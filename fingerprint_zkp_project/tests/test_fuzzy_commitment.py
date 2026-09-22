"""
Unit tests for Fuzzy Commitment Template Protection and ECC modules.
"""

import numpy as np
from template_protection import generate_secret_key, BlockECC, FuzzyCommitment

def test_secret_key_generation():
    bytes_val, bits_val = generate_secret_key(bit_length=64)
    assert len(bytes_val) == 8
    assert len(bits_val) == 64
    assert set(np.unique(bits_val)).issubset({0, 1})

def test_block_ecc():
    ecc = BlockECC(message_bits=64, codeword_bits=256)
    secret = np.random.choice([0, 1], size=64).astype(np.uint8)
    codeword = ecc.encode(secret)
    assert len(codeword) == 256

    # Burst noise in a handful of codeword bits.
    noisy = codeword.copy()
    noisy[0:5] = 1 - noisy[0:5]

    recovered, corrected, success = ecc.decode(noisy)
    assert success is True
    assert np.array_equal(recovered, secret)

    # Destroy one logical replica (64 interleaved positions). Remaining copies vote.
    noisy_replica = codeword.copy()
    replica0 = ecc.inv[:64]
    noisy_replica[replica0] = 1 - noisy_replica[replica0]
    recovered2, _, _ = ecc.decode(noisy_replica)
    assert np.array_equal(recovered2, secret)

def test_fuzzy_commitment_enrollment_and_recovery():
    fc = FuzzyCommitment(secret_bits=64, vector_bits=256)
    b_enrolled = np.random.choice([0, 1], size=256).astype(np.uint8)

    enrollment = fc.enroll(b_enrolled)
    helper = enrollment["helper_data"]
    commitment = enrollment["commitment"]

    # 1. Exact match query
    res_exact = fc.recover_secret(b_enrolled, helper, commitment)
    assert res_exact["success"] is True
    assert np.array_equal(res_exact["recovered_secret"], enrollment["secret"])

    # 2. Query with intra-user noise plus a residual polar-sector rotation
    b_noisy = b_enrolled.copy()
    b_noisy[[0, 4, 8, 12, 16]] = 1 - b_noisy[[0, 4, 8, 12, 16]]
    b_rotated = np.roll(b_noisy, 16)
    res_noisy = fc.recover_secret(b_rotated, helper, commitment)
    assert res_noisy["success"] is True

    # 3. Impostor query with completely different binary vector
    b_impostor = np.random.choice([0, 1], size=256).astype(np.uint8)
    res_impostor = fc.recover_secret(b_impostor, helper, commitment)
    assert res_impostor["success"] is False
