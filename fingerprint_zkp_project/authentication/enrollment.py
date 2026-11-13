"""
Fingerprint Enrollment Module.
Integrates 384-bit annular feature extraction, Fuzzy Commitment helper generation,
and Ristretto255 Schnorr public key registration.
"""

import os
import sys
import secrets
from typing import Tuple, Optional, Union
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import _quadrature_gabor_bank, N_FILTERS
from crypto.schnorr import SchnorrProver, Ristretto255Group
from crypto.key_derivation import derive_witness
from authentication.database import EnrollmentRecord, ClientCredential

# Dominant orientation weights from development set (Gray-b2, Gray-b1, Gray-b0)
DEV_WEIGHTS_3CH = np.array([1.06450, 1.28243, 0.85054], dtype=np.float64)

# Fixed, reproducible public interleaver permutation for N=384, K=32, R=12
def get_public_interleaver(seed: int = 20261112) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(384).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    return perm, inv

def extract_384_template_features(img_input: Union[str, np.ndarray, dict]) -> dict:
    """
    Extracts the audited 384-bit annular orientation template along with full
    intermediate cell coverage, dominant orientations, and quality metrics.
    """
    if isinstance(img_input, dict) and "vec_384" in img_input and "mask_384" in img_input:
        return img_input

    if isinstance(img_input, str):
        img = cv2.imread(img_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image at {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")

    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)

    prepared = tmpl._prepare_polar(img)
    if prepared is None:
        raise ValueError("Polar region preparation failed.")
    img_w, mask_w, ori_w, center, max_r, inner = prepared

    cx, cy = center
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    # Step 7 Radial Sampling Grid over [inner, max_r]
    # 256 angular steps x 64 radial steps -> 128 area samples per (16x8) polar cell
    n_th, n_r = 256, 64
    thetas = np.linspace(0, 2 * np.pi, n_th, endpoint=False).astype(np.float32)
    radii = (inner + (np.arange(n_r, dtype=np.float32) + 0.5) * (max_r - inner) / float(n_r))
    R_grid, TH_grid = np.meshgrid(radii, thetas)
    map_x = (cx + R_grid * np.cos(TH_grid)).astype(np.float32)
    map_y = (cy + R_grid * np.sin(TH_grid)).astype(np.float32)

    gabor_bank = _quadrature_gabor_bank()
    p_energies = []
    filter_thetas = []
    for ke, ko, th_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        e_hi = cv2.remap(e, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        p_energies.append(cv2.resize(e_hi, (8, 16), interpolation=cv2.INTER_AREA))
        filter_thetas.append(th_k)

    m_hi = cv2.remap(ring_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    polar_m = cv2.resize(m_hi, (8, 16), interpolation=cv2.INTER_AREA)

    # Dominant Gabor Orientation via circular vector summation
    U = np.sum([p_energies[k] * np.cos(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    V = np.sum([p_energies[k] * np.sin(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_gabor = np.mod(0.5 * np.arctan2(V, U), np.pi).astype(np.float32)

    q = np.clip(np.floor(8.0 * theta_gabor / np.pi).astype(np.int32), 0, 7)
    g = q ^ (q >> 1) # 3-bit circular Gray code

    # Assemble 384-bit representation (128 cells x 3 Gray bits)
    v384 = []
    m384 = []
    for s in range(16):
        for r_idx in range(8):
            c_cov = float(polar_m[s, r_idx])
            c_mask = 1 if c_cov >= 0.70 else 0
            g_val = int(g[s, r_idx])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1

            v384.extend([b2, b1, b0])
            m384.extend([c_mask, c_mask, c_mask])

    vec_384 = np.array(v384, dtype=np.uint8)
    mask_384 = np.array(m384, dtype=np.uint8)
    quality = float(np.mean(mask_384))

    return {
        "vec_384": vec_384,
        "mask_384": mask_384,
        "polar_m": polar_m,
        "theta_gabor": theta_gabor,
        "quality": quality
    }

def extract_384_template(img_input: Union[str, np.ndarray, dict, Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the audited 384-bit annular orientation template and coverage mask.
    Returns:
        (vec_384: np.ndarray uint8, mask_384: np.ndarray uint8)
    """
    if isinstance(img_input, (tuple, list)) and len(img_input) == 2 and isinstance(img_input[0], np.ndarray):
        return np.asarray(img_input[0], dtype=np.uint8), np.asarray(img_input[1], dtype=np.uint8)
    res = extract_384_template_features(img_input)
    return res["vec_384"], res["mask_384"]

def align_and_fuse_5_impressions(
    images: list
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Performs rotational alignment, quality calculation, and reliability-weighted
    template fusion across 5 fingerprint impressions to produce ONE 384-bit template.

    Args:
        images: List or tuple of exactly 5 fingerprint images (paths or numpy arrays).

    Returns:
        (fused_vec: np.ndarray uint8, fused_mask: np.ndarray uint8, meta: dict)
    """
    if len(images) != 5:
        raise ValueError(f"Multi-image enrollment requires exactly 5 impressions, got {len(images)}")

    # 1. Feature Extraction on all 5 images independently
    features = [extract_384_template_features(img) for img in images]
    qualities = [f["quality"] for f in features]

    if max(qualities) < 0.40:
        raise ValueError("Enrollment impressions quality insufficient (all impressions < 40% active mask coverage).")

    # 2. Select Reference Impression (highest quality / mask coverage)
    ref_idx = int(np.argmax(qualities))
    v_ref = features[ref_idx]["vec_384"]
    m_ref = features[ref_idx]["mask_384"]

    # 3. Rotational Alignment of impressions k != ref to ref across 16 cyclic sector shifts
    bps = 24 # 24 bits per angular sector (8 radial cells x 3 Gray bits)
    w_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    v_aligned = []
    m_aligned = []
    cov_aligned = []
    align_shifts = []
    align_agreements = []

    for k in range(5):
        if k == ref_idx:
            v_aligned.append(v_ref)
            m_aligned.append(m_ref)
            cov_aligned.append(features[k]["polar_m"])
            align_shifts.append(0)
            align_agreements.append(1.0)
            continue

        best_s = 0
        best_agree = -1.0
        v_k = features[k]["vec_384"]
        m_k = features[k]["mask_384"]

        for s in range(16):
            sh_v = np.roll(v_k, s * bps)
            sh_m = np.roll(m_k, s * bps)
            mut_m = (m_ref > 0) & (sh_m > 0)
            n_mut = np.sum(mut_m)
            if n_mut < 72: # at least 3 sectors overlap
                continue
            # Weighted bit agreement score
            agree = np.sum((v_ref == sh_v) * mut_m * w_384) / np.sum(mut_m * w_384)
            if agree > best_agree:
                best_agree = agree
                best_s = s

        align_shifts.append(best_s)
        align_agreements.append(float(best_agree) if best_agree >= 0 else 0.50)
        v_aligned.append(np.roll(v_k, best_s * bps))
        m_aligned.append(np.roll(m_k, best_s * bps))
        cov_aligned.append(np.roll(features[k]["polar_m"], best_s, axis=0))

    # 4. Mask-aware, reliability-weighted fusion
    t_fused = np.zeros(384, dtype=np.uint8)
    m_fused = np.zeros(384, dtype=np.uint8)

    for j in range(384):
        c_idx = j // 3 # cell index 0..127
        s_cell = c_idx // 8
        r_cell = c_idx % 8
        w_ch = DEV_WEIGHTS_3CH[j % 3]

        score = 0.0
        n_obs = 0
        ref_obs = (m_aligned[ref_idx][j] > 0)

        for k in range(5):
            if m_aligned[k][j] > 0:
                n_obs += 1
                c_cov = float(cov_aligned[k][s_cell, r_cell])
                # Combined impression quality and cell reliability weight
                w_kj = qualities[k] * max(0.50, c_cov) * w_ch
                sgn = 2.0 * float(v_aligned[k][j]) - 1.0
                score += w_kj * sgn

        # Majority vote with tie-break to reference impression
        if score > 1e-6:
            t_fused[j] = 1
        elif score < -1e-6:
            t_fused[j] = 0
        else:
            t_fused[j] = v_aligned[ref_idx][j]

        # Mask-aware consensus: active if observed in >= 2 impressions or in reference
        m_fused[j] = 1 if (n_obs >= 2 or ref_obs) else 0

    fusion_meta = {
        "ref_impression_idx": ref_idx,
        "qualities": qualities,
        "align_shifts": align_shifts,
        "align_agreements": align_agreements,
        "fused_coverage": float(np.mean(m_fused)),
        "ref_coverage": float(qualities[ref_idx])
    }

    return t_fused, m_fused, fusion_meta

def enroll_user(
    user_id: str,
    fingerprint_img: Union[str, np.ndarray, list, tuple],
    credential_K: Optional[bytes] = None,
    seed: int = 20261112
) -> Tuple[EnrollmentRecord, ClientCredential]:
    """
    Executes secure fingerprint enrollment.
    Supports single-image enrollment (str / np.ndarray) and 5-image fused enrollment (list / tuple).
    
    Args:
        user_id: Unique user identifier.
        fingerprint_img: Single fingerprint image or list of 5 fingerprint impressions.
        credential_K: Optional client credential (256 bits). If None, generates fresh CSPRNG token.
        seed: Public interleaver seed.
        
    Returns:
        Tuple of (EnrollmentRecord for server, ClientCredential for user device).
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string.")

    # 1. Feature Extraction (384 bits): single image, pre-extracted template, or 5-image fusion
    if isinstance(fingerprint_img, dict) and "vec_384" in fingerprint_img:
        t_enroll, m_enroll = fingerprint_img["vec_384"], fingerprint_img["mask_384"]
    elif isinstance(fingerprint_img, (list, tuple)) and len(fingerprint_img) == 2 and isinstance(fingerprint_img[0], np.ndarray):
        t_enroll, m_enroll = np.asarray(fingerprint_img[0], dtype=np.uint8), np.asarray(fingerprint_img[1], dtype=np.uint8)
    elif isinstance(fingerprint_img, (list, tuple)):
        if len(fingerprint_img) == 5:
            t_enroll, m_enroll, _ = align_and_fuse_5_impressions(fingerprint_img)
        elif len(fingerprint_img) == 1:
            t_enroll, m_enroll = extract_384_template(fingerprint_img[0])
        else:
            raise ValueError(f"Multi-image enrollment expects 1 or 5 impressions, got {len(fingerprint_img)}")
    else:
        t_enroll, m_enroll = extract_384_template(fingerprint_img)

    if float(np.mean(m_enroll)) < 0.50:
        raise ValueError("Enrollment fingerprint quality insufficient (active mask coverage < 50%).")

    # 2. Sample 32-bit biometric secret S
    # Cryptographically secure random 32-bit secret
    secret_int = secrets.randbits(32)
    secret_bits = np.array([(secret_int >> i) & 1 for i in range(32)], dtype=np.uint8)

    # 3. Interleaved Repetition Encoding (R=12, N=384, K=32)
    perm, _ = get_public_interleaver(seed)
    codeword = np.tile(secret_bits, 12)[perm]

    # 4. Helper Data W = T_enroll XOR Codeword(S)
    helper_bits = np.bitwise_xor(t_enroll, codeword)

    # 5. Independent High-Entropy Credential K (256 bits)
    if credential_K is None:
        credential_K = secrets.token_bytes(32)
    elif len(credential_K) < 16:
        raise ValueError("Client credential K must be at least 16 bytes (32 bytes recommended).")

    # 6. User Salt (16 bytes)
    user_salt = secrets.token_bytes(16)

    # 7. Two-Factor Witness Derivation x = HKDF(S || K, salt) mod L
    witness_x = derive_witness(secret_bits, credential_K, user_salt)

    # 8. Schnorr Public Key Y = x * B
    prover = SchnorrProver(witness_x)
    public_key_Y = prover.public_key

    # 9. Pack bit vectors to bytes for compact storage
    record = EnrollmentRecord(
        user_id=user_id,
        helper_data=np.packbits(helper_bits).tobytes(),
        mask=np.packbits(m_enroll).tobytes(),
        salt=user_salt,
        public_key_Y=public_key_Y
    )

    client_cred = ClientCredential(
        user_id=user_id,
        credential_K=credential_K
    )

    return record, client_cred

def enroll_user_multi(
    user_id: str,
    fingerprint_imgs: list,
    credential_K: Optional[bytes] = None,
    seed: int = 20261112
) -> Tuple[EnrollmentRecord, ClientCredential, dict]:
    """
    Dedicated multi-impression enrollment pipeline using exactly 5 impressions.
    Returns (EnrollmentRecord, ClientCredential, fusion_meta).
    """
    if len(fingerprint_imgs) != 5:
        raise ValueError(f"enroll_user_multi requires exactly 5 impressions, got {len(fingerprint_imgs)}")

    t_fused, m_fused, fusion_meta = align_and_fuse_5_impressions(fingerprint_imgs)

    record, cred = enroll_user(
        user_id=user_id,
        fingerprint_img=fingerprint_imgs,
        credential_K=credential_K,
        seed=seed
    )

    return record, cred, fusion_meta
