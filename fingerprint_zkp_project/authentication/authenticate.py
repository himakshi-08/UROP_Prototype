"""
Fingerprint Authentication Module.
Executes the end-to-end authentication workflow:
1. Feature extraction of query fingerprint (384-bit annular orientation).
2. Ciphertext-guided cyclic alignment across 16 sector shifts.
3. Erasure-aware LLR-weighted majority decoding to recover biometric secret S'.
4. Two-factor witness reconstruction x' = HKDF(S' || K, salt) mod L.
5. Interactive 3-pass Ristretto255 Schnorr Zero-Knowledge Proof.
6. Server-side session verification against registered public key Y.
"""

import os
import sys
import time
from typing import Tuple, Optional, Union, Dict
from dataclasses import dataclass
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Development-trained LLR orientation weights strictly estimated on fp_training (task-1124)
# Gray-b2: 1.06450, Gray-b1: 1.28243, Gray-b0: 0.85054
DEV_WEIGHTS_3CH = np.array([1.06450, 1.28243, 0.85054], dtype=np.float64)
from crypto.schnorr import SchnorrProver, Ristretto255Group
from crypto.key_derivation import derive_witness
from crypto.zkp_protocol import ZKPSessionManager
from authentication.database import EnrollmentRecord
from authentication.enrollment import extract_384_template, get_public_interleaver

@dataclass
class AuthenticationResult:
    is_authenticated: bool
    secret_recovered: bool
    zkp_verified: bool
    status_message: str
    optimal_shift: int
    num_valid_bits: int
    overlap_ratio: float
    latency_ms: Dict[str, float]

def decode_fuzzy_secret(
    query_vec: np.ndarray,
    query_mask: np.ndarray,
    helper_bits: np.ndarray,
    enroll_mask: np.ndarray,
    inv_perm: np.ndarray,
    weights_384: np.ndarray,
    min_overlap: float = 0.50
) -> Tuple[Optional[np.ndarray], int, int, float, str]:
    """
    Performs ciphertext-guided alignment and erasure-aware LLR-weighted majority decoding.
    
    Returns:
        (recovered_secret, optimal_shift, n_valid, overlap, failure_reason)
    """
    k_bits = 32
    rep_factor = 12
    bps = 24

    best_s = 0
    best_score = -1e9

    # 1. Ciphertext-Guided Cyclic Alignment Search
    # Evaluates weighted replica consistency across all 16 shifts
    for s in range(16):
        sh_q = np.roll(query_vec, s * bps)
        sh_m = np.roll(query_mask, s * bps)
        mut_m = (enroll_mask > 0) & (sh_m > 0)
        
        n_mut = int(np.sum(mut_m))
        if n_mut / 384.0 < min_overlap:
            continue
            
        cand_cw = np.bitwise_xor(helper_bits, sh_q)
        deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
        deint_m = mut_m[inv_perm].reshape(rep_factor, k_bits)
        deint_w = weights_384[inv_perm].reshape(rep_factor, k_bits)

        score = 0.0
        for j in range(k_bits):
            col_b = deint_cw[:, j]
            col_m = deint_m[:, j]
            col_w = deint_w[:, j]
            val = (col_m > 0)
            if np.any(val):
                sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
                score += abs(float(np.sum(col_w[val] * sgn)))

        if score > best_score:
            best_score = score
            best_s = s

    # 2. Decode at optimal shift s*
    sh_q_opt = np.roll(query_vec, best_s * bps)
    sh_m_opt = np.roll(query_mask, best_s * bps)
    mut_m_opt = (enroll_mask > 0) & (sh_m_opt > 0)
    n_valid_opt = int(np.sum(mut_m_opt))
    overlap_opt = float(n_valid_opt / 384.0)

    if overlap_opt < min_overlap:
        return None, best_s, n_valid_opt, overlap_opt, "Mutual template overlap below 50% threshold."

    cand_cw_opt = np.bitwise_xor(helper_bits, sh_q_opt)
    deint_cw = cand_cw_opt[inv_perm].reshape(rep_factor, k_bits)
    deint_m = mut_m_opt[inv_perm].reshape(rep_factor, k_bits)
    deint_w = weights_384[inv_perm].reshape(rep_factor, k_bits)

    recovered_bits = []
    for j in range(k_bits):
        col_b = deint_cw[:, j]
        col_m = deint_m[:, j]
        col_w = deint_w[:, j]
        val = (col_m > 0)
        
        # Complete block erasure
        if not np.any(val):
            return None, best_s, n_valid_opt, overlap_opt, f"Bit {j} completely erased (all 12 replicas unobserved)."
            
        sgn = 2.0 * col_b[val].astype(np.float64) - 1.0
        sc = float(np.sum(col_w[val] * sgn))
        
        # Strict tie classification (never coerced)
        if sc > 1e-6:
            recovered_bits.append(1)
        elif sc < -1e-6:
            recovered_bits.append(0)
        else:
            return None, best_s, n_valid_opt, overlap_opt, f"Replica tie encountered on secret bit {j} (ambiguous)."

    return np.array(recovered_bits, dtype=np.uint8), best_s, n_valid_opt, overlap_opt, ""

def authenticate_user(
    user_id: str,
    query_fingerprint_img: Union[str, np.ndarray],
    credential_K: bytes,
    record: EnrollmentRecord,
    session_manager: ZKPSessionManager,
    seed: int = 20261112
) -> AuthenticationResult:
    """
    Executes complete end-to-end authentication workflow.
    """
    t_start = time.perf_counter()
    latency = {}

    if record.user_id != user_id:
        return AuthenticationResult(
            is_authenticated=False,
            secret_recovered=False,
            zkp_verified=False,
            status_message="User ID mismatch with enrollment record.",
            optimal_shift=0,
            num_valid_bits=0,
            overlap_ratio=0.0,
            latency_ms={"total": 0.0}
        )

    # Step 1: Feature Extraction
    t0 = time.perf_counter()
    query_vec, query_mask = extract_384_template(query_fingerprint_img)
    latency["feature_extraction"] = (time.perf_counter() - t0) * 1000.0

    # Step 2: Fuzzy Commitment Decoding
    t0 = time.perf_counter()
    helper_bits = record.get_helper_bits()
    enroll_mask = record.get_mask_bits()
    _, inv_perm = get_public_interleaver(seed)
    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    recovered_S, opt_s, n_val, overlap, fail_reason = decode_fuzzy_secret(
        query_vec, query_mask, helper_bits, enroll_mask, inv_perm, weights_384
    )
    latency["ecc_decoding"] = (time.perf_counter() - t0) * 1000.0

    if recovered_S is None:
        latency["total"] = (time.perf_counter() - t_start) * 1000.0
        return AuthenticationResult(
            is_authenticated=False,
            secret_recovered=False,
            zkp_verified=False,
            status_message=f"Biometric secret recovery failed: {fail_reason}",
            optimal_shift=opt_s,
            num_valid_bits=n_val,
            overlap_ratio=overlap,
            latency_ms=latency
        )

    # Step 3: Two-Factor Witness Derivation x' = HKDF(S' || K, salt) mod L
    t0 = time.perf_counter()
    try:
        witness_x_prime = derive_witness(recovered_S, credential_K, record.salt)
        prover = SchnorrProver(witness_x_prime)
    except Exception as e:
        latency["total"] = (time.perf_counter() - t_start) * 1000.0
        return AuthenticationResult(
            is_authenticated=False,
            secret_recovered=True,
            zkp_verified=False,
            status_message=f"Witness derivation error: {str(e)}",
            optimal_shift=opt_s,
            num_valid_bits=n_val,
            overlap_ratio=overlap,
            latency_ms=latency
        )
    latency["witness_derivation"] = (time.perf_counter() - t0) * 1000.0

    # Step 4: Interactive Schnorr Sigma Protocol
    t0 = time.perf_counter()
    # Pass 1: Prover Commit
    T = prover.commit()

    # Pass 2: Server Challenge
    session_id, challenge_c = session_manager.create_session(user_id, record.public_key_Y, T)

    # Pass 3: Prover Respond
    response_z = prover.respond(challenge_c)

    # Pass 4: Server Verify
    is_valid, msg = session_manager.verify_response(session_id, response_z)
    latency["zkp_protocol"] = (time.perf_counter() - t0) * 1000.0
    latency["total"] = (time.perf_counter() - t_start) * 1000.0

    return AuthenticationResult(
        is_authenticated=is_valid,
        secret_recovered=True,
        zkp_verified=is_valid,
        status_message=msg,
        optimal_shift=opt_s,
        num_valid_bits=n_val,
        overlap_ratio=overlap,
        latency_ms=latency
    )
