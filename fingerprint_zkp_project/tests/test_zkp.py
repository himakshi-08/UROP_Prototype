"""
Unit tests for Schnorr Zero-Knowledge Proof protocol.
"""

import numpy as np
from zkp import SchnorrGroup, UserDeviceProver, ServerVerifier

def test_schnorr_zkp_completeness_and_soundness():
    group = SchnorrGroup()
    secret = np.random.choice([0, 1], size=64).astype(np.uint8)

    # Prover & Verifier setup
    prover = UserDeviceProver(secret, group=group)
    verifier = ServerVerifier(prover.public_key, group=group)

    # Step 1: Commitment
    t = prover.step1_create_commitment()

    # Step 2: Challenge
    c = verifier.step1_issue_challenge()

    # Step 3: Response
    s = prover.step2_respond_to_challenge(c)

    # Step 4: Verification equation g^s == t * y^c mod p
    valid = verifier.step2_verify(t, s, challenge_c=c)
    assert valid is True

    # Test Soundness: Tampered response should fail verification
    invalid = verifier.step2_verify(t, (s + 1) % group.q, challenge_c=c)
    assert invalid is False
