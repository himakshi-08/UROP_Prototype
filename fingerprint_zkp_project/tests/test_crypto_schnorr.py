"""
Comprehensive Unit & Security Test Suite for Ristretto255 Cryptographic Module.
Tests:
1. Ristretto255 Group arithmetic (addition, scalar multiplication, point validation).
2. Schnorr Sigma Protocol correctness and completeness.
3. Cryptographic soundness: tampered responses, commitments, and wrong witnesses.
4. Two-factor HKDF-SHA256 witness derivation and domain separation.
5. Session-bound challenge management, expiration, and replay prevention.
6. Nonce reuse and small-subgroup safety.
"""

import os
import sys
import time
import secrets
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from crypto import (
    Ristretto255Group,
    SchnorrProver,
    SchnorrVerifier,
    derive_witness,
    ZKPSessionManager,
    POINT_BYTES,
    SCALAR_BYTES
)

# ---------------------------------------------------------------------------
# 1. Ristretto255 Group Arithmetic Tests
# ---------------------------------------------------------------------------
def test_valid_point_generation():
    s = Ristretto255Group.random_scalar()
    P = Ristretto255Group.base_scalarmult(s)
    assert len(P) == POINT_BYTES
    assert Ristretto255Group.is_valid_point(P) is True

def test_invalid_point_rejection():
    # 32 bytes of 0xFF is an invalid Ristretto255 encoding
    bad_point = b"\xff" * 32
    assert Ristretto255Group.is_valid_point(bad_point) is False
    with pytest.raises(ValueError):
        Ristretto255Group.scalarmult(Ristretto255Group.random_scalar(), bad_point)

def test_point_addition_commutativity():
    s1 = Ristretto255Group.random_scalar()
    s2 = Ristretto255Group.random_scalar()
    P1 = Ristretto255Group.base_scalarmult(s1)
    P2 = Ristretto255Group.base_scalarmult(s2)
    
    P_12 = Ristretto255Group.point_add(P1, P2)
    P_21 = Ristretto255Group.point_add(P2, P1)
    assert P_12 == P_21

def test_scalar_multiplication_linearity():
    # (a + b) * B == a * B + b * B
    a = Ristretto255Group.random_scalar()
    b = Ristretto255Group.random_scalar()
    ab = Ristretto255Group.scalar_add(a, b)
    
    lhs = Ristretto255Group.base_scalarmult(ab)
    rhs = Ristretto255Group.point_add(
        Ristretto255Group.base_scalarmult(a),
        Ristretto255Group.base_scalarmult(b)
    )
    assert lhs == rhs

def test_scalar_multiplication_associativity():
    # (a * b) * B == a * (b * B)
    a = Ristretto255Group.random_scalar()
    b = Ristretto255Group.random_scalar()
    ab = Ristretto255Group.scalar_mul(a, b)
    
    lhs = Ristretto255Group.base_scalarmult(ab)
    b_B = Ristretto255Group.base_scalarmult(b)
    rhs = Ristretto255Group.scalarmult(a, b_B)
    assert lhs == rhs

# ---------------------------------------------------------------------------
# 2. Schnorr Sigma Protocol Correctness & Soundness Tests
# ---------------------------------------------------------------------------
def test_schnorr_protocol_round_trip():
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)
    verifier = SchnorrVerifier(prover.public_key)

    T = prover.commit()
    c = verifier.generate_challenge()
    z = prover.respond(c)

    assert verifier.verify(T, c, z) is True

def test_tampered_response_rejection():
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)
    verifier = SchnorrVerifier(prover.public_key)

    T = prover.commit()
    c = verifier.generate_challenge()
    z = prover.respond(c)

    # Tamper with response scalar
    bad_z = bytes([z[0] ^ 0x01]) + z[1:]
    assert verifier.verify(T, c, bad_z) is False

def test_tampered_commitment_rejection():
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)
    verifier = SchnorrVerifier(prover.public_key)

    T = prover.commit()
    c = verifier.generate_challenge()
    z = prover.respond(c)

    # Use a different valid commitment
    other_T = Ristretto255Group.base_scalarmult(Ristretto255Group.random_scalar())
    assert verifier.verify(other_T, c, z) is False

def test_tampered_challenge_rejection():
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)
    verifier = SchnorrVerifier(prover.public_key)

    T = prover.commit()
    c = verifier.generate_challenge()
    z = prover.respond(c)

    # Verifier checks with a different challenge
    different_c = Ristretto255Group.random_scalar()
    assert verifier.verify(T, different_c, z) is False

def test_wrong_witness_rejection():
    x1 = Ristretto255Group.random_scalar()
    x2 = Ristretto255Group.random_scalar()
    
    prover1 = SchnorrProver(x1)
    verifier2 = SchnorrVerifier(Ristretto255Group.base_scalarmult(x2))

    T = prover1.commit()
    c = verifier2.generate_challenge()
    z = prover1.respond(c)

    assert verifier2.verify(T, c, z) is False

def test_prover_nonce_reuse_prevention():
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)
    
    prover.commit()
    c = Ristretto255Group.random_scalar()
    prover.respond(c)

    # Calling respond again without a new commit must fail immediately
    with pytest.raises(RuntimeError):
        prover.respond(c)

# ---------------------------------------------------------------------------
# 3. Two-Factor HKDF Key Derivation Tests
# ---------------------------------------------------------------------------
def test_hkdf_witness_derivation_determinism():
    S = b"\x12\x34\x56\x78"
    K = os.urandom(32)
    salt = os.urandom(16)

    w1 = derive_witness(S, K, salt)
    w2 = derive_witness(S, K, salt)
    assert len(w1) == SCALAR_BYTES
    assert w1 == w2

def test_hkdf_witness_sensitivity_to_secret():
    S1 = b"\x12\x34\x56\x78"
    S2 = b"\x12\x34\x56\x79" # 1 bit difference
    K = os.urandom(32)
    salt = os.urandom(16)

    w1 = derive_witness(S1, K, salt)
    w2 = derive_witness(S2, K, salt)
    assert w1 != w2

def test_hkdf_witness_sensitivity_to_credential():
    S = b"\x12\x34\x56\x78"
    K1 = os.urandom(32)
    K2 = os.urandom(32)
    salt = os.urandom(16)

    w1 = derive_witness(S, K1, salt)
    w2 = derive_witness(S, K2, salt)
    assert w1 != w2

def test_hkdf_context_separation():
    S = b"\x12\x34\x56\x78"
    K = os.urandom(32)
    salt = os.urandom(16)

    w_v1 = derive_witness(S, K, salt, context=b"Context-A")
    w_v2 = derive_witness(S, K, salt, context=b"Context-B")
    assert w_v1 != w_v2

def test_hkdf_numpy_secret_array_compatibility():
    # Test array of 32 binary bits [0, 1, 1, 0, ...]
    bit_arr = np.random.randint(0, 2, 32, dtype=np.uint8)
    packed_bytes = np.packbits(bit_arr).tobytes()
    K = os.urandom(32)
    salt = os.urandom(16)

    w_arr = derive_witness(bit_arr, K, salt)
    w_bytes = derive_witness(packed_bytes, K, salt)
    assert w_arr == w_bytes

def test_hkdf_short_credential_rejection():
    S = b"\x12\x34\x56\x78"
    short_K = b"short_pin" # < 16 bytes
    salt = os.urandom(16)
    with pytest.raises(ValueError):
        derive_witness(S, short_K, salt)

# ---------------------------------------------------------------------------
# 4. Interactive ZKP Session & Replay Protection Tests
# ---------------------------------------------------------------------------
def test_session_manager_valid_flow():
    mgr = ZKPSessionManager(default_ttl=10.0)
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)

    T = prover.commit()
    sid, c = mgr.create_session("alice", prover.public_key, T)
    z = prover.respond(c)

    success, msg = mgr.verify_response(sid, z)
    assert success is True
    assert "successfully" in msg

def test_replay_attack_prevention():
    mgr = ZKPSessionManager(default_ttl=10.0)
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)

    T = prover.commit()
    sid, c = mgr.create_session("alice", prover.public_key, T)
    z = prover.respond(c)

    # 1. First verification succeeds
    ok1, _ = mgr.verify_response(sid, z)
    assert ok1 is True

    # 2. Replay attempt using same session_id MUST fail
    ok2, msg2 = mgr.verify_response(sid, z)
    assert ok2 is False
    assert "already consumed" in msg2.lower() or "invalid" in msg2.lower()

def test_session_expiration():
    # Very short TTL
    mgr = ZKPSessionManager(default_ttl=0.1)
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)

    T = prover.commit()
    sid, c = mgr.create_session("alice", prover.public_key, T, ttl_seconds=0.1)
    z = prover.respond(c)

    # Wait for session to expire
    time.sleep(0.15)

    ok, msg = mgr.verify_response(sid, z)
    assert ok is False
    assert "expired" in msg.lower() or "invalid" in msg.lower()

def test_nonce_deduplication_across_sessions():
    mgr = ZKPSessionManager(default_ttl=10.0)
    x = Ristretto255Group.random_scalar()
    prover = SchnorrProver(x)

    # Fixed commitment point T
    T = prover.commit()
    sid1, c1 = mgr.create_session("alice", prover.public_key, T)

    # Attacker tries to reuse the exact same commitment T in a new session
    with pytest.raises(ValueError, match="Nonce reuse detected"):
        mgr.create_session("alice", prover.public_key, T)

def test_concurrent_sessions_isolation():
    mgr = ZKPSessionManager()
    
    # Alice session
    x_a = Ristretto255Group.random_scalar()
    prover_a = SchnorrProver(x_a)
    T_a = prover_a.commit()
    sid_a, c_a = mgr.create_session("alice", prover_a.public_key, T_a)
    z_a = prover_a.respond(c_a)

    # Bob session
    x_b = Ristretto255Group.random_scalar()
    prover_b = SchnorrProver(x_b)
    T_b = prover_b.commit()
    sid_b, c_b = mgr.create_session("bob", prover_b.public_key, T_b)
    z_b = prover_b.respond(c_b)

    # Alice's response to Bob's session must fail
    ok_cross, _ = mgr.verify_response(sid_b, z_a)
    assert ok_cross is False

    # Alice's response to Alice's session must succeed
    ok_a, _ = mgr.verify_response(sid_a, z_a)
    assert ok_a is True

# ---------------------------------------------------------------------------
# 5. Integrated Two-Factor End-to-End Tests
# ---------------------------------------------------------------------------
def test_full_two_factor_authentication_success():
    # Enrollment
    S = secrets.token_bytes(4)
    K = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    
    # Enrolled private witness and public key
    x_enroll = derive_witness(S, K, salt)
    prover = SchnorrProver(x_enroll)
    Y_registered = prover.public_key

    # Authentication Phase (Claimant has correct S and correct K)
    S_recovered = S
    K_presented = K
    x_auth = derive_witness(S_recovered, K_presented, salt)
    auth_prover = SchnorrProver(x_auth)

    mgr = ZKPSessionManager()
    T = auth_prover.commit()
    sid, c = mgr.create_session("user_101", Y_registered, T)
    z = auth_prover.respond(c)

    success, msg = mgr.verify_response(sid, z)
    assert success is True

def test_two_factor_authentication_impostor_fingerprint_rejection():
    # Victim enrolled
    S_victim = b"\xaa\xbb\xcc\xdd"
    K_victim = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    
    x_victim = derive_witness(S_victim, K_victim, salt)
    Y_registered = SchnorrProver(x_victim).public_key

    # Impostor has different biometric secret S_imp, even if they have stolen K
    S_impostor = b"\xaa\xbb\xcc\xee" # 1 bit difference
    x_impostor = derive_witness(S_impostor, K_victim, salt)
    imp_prover = SchnorrProver(x_impostor)

    mgr = ZKPSessionManager()
    T = imp_prover.commit()
    sid, c = mgr.create_session("user_101", Y_registered, T)
    z = imp_prover.respond(c)

    success, _ = mgr.verify_response(sid, z)
    assert success is False

def test_two_factor_authentication_wrong_credential_rejection():
    # Victim enrolled
    S = b"\xaa\xbb\xcc\xdd"
    K_victim = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    
    x_victim = derive_witness(S, K_victim, salt)
    Y_registered = SchnorrProver(x_victim).public_key

    # Attacker matches biometric (25% impostor recovery!), but lacks K
    K_wrong = secrets.token_bytes(32)
    x_attacker = derive_witness(S, K_wrong, salt)
    att_prover = SchnorrProver(x_attacker)

    mgr = ZKPSessionManager()
    T = att_prover.commit()
    sid, c = mgr.create_session("user_101", Y_registered, T)
    z = att_prover.respond(c)

    success, _ = mgr.verify_response(sid, z)
    assert success is False

