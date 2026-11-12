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

# Fixed, reproducible public interleaver permutation for N=384, K=32, R=12
def get_public_interleaver(seed: int = 20261112) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(384).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    return perm, inv

def extract_384_template(img_input: Union[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the audited 384-bit annular orientation template and coverage mask
    using explicit coordinate remap over [0.18 * r_max, r_max] (128 cells x 3 Gray bits).
    Returns:
        (vec_384: np.ndarray uint8, mask_384: np.ndarray uint8)
    """
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

    return np.array(v384, dtype=np.uint8), np.array(m384, dtype=np.uint8)

def enroll_user(
    user_id: str,
    fingerprint_img: Union[str, np.ndarray],
    credential_K: Optional[bytes] = None,
    seed: int = 20261112
) -> Tuple[EnrollmentRecord, ClientCredential]:
    """
    Executes secure fingerprint enrollment.
    
    Args:
        user_id: Unique user identifier.
        fingerprint_img: Fingerprint image path or grayscale numpy array.
        credential_K: Optional client credential (256 bits). If None, generates fresh CSPRNG token.
        seed: Public interleaver seed.
        
    Returns:
        Tuple of (EnrollmentRecord for server, ClientCredential for user device).
    """
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string.")

    # 1. Feature Extraction (384 bits)
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
