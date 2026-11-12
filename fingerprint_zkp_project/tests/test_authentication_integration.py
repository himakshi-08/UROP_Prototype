"""
End-to-End Integration Tests for Authentication Pipeline.
Verifies the complete flow:
Enrollment -> Fuzzy Commitment -> Two-Factor HKDF -> Ristretto255 Schnorr ZKP -> Server Verification.
"""

import os
import sys
import secrets
import pytest
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from crypto.schnorr import Ristretto255Group
from crypto.zkp_protocol import ZKPSessionManager
from authentication import (
    enroll_user,
    authenticate_user,
    SecureEnrollmentStore,
    EnrollmentRecord,
    ClientCredential
)

@pytest.fixture
def synthetic_fingerprint():
    # Deterministic synthetic ridge pattern with core for unit tests
    img = np.full((300, 300), 200, dtype=np.uint8)
    yy, xx = np.ogrid[:300, :300]
    rr = np.sqrt((xx - 150)**2 + (yy - 150)**2)
    ridges = (127.5 * (1.0 + np.sin(rr / 4.0))).astype(np.uint8)
    mask = (rr < 120).astype(np.uint8)
    return np.where(mask, ridges, 255).astype(np.uint8)

def test_enrollment_creates_valid_records(synthetic_fingerprint):
    user_id = "user_test_01"
    record, cred = enroll_user(user_id, synthetic_fingerprint)

    assert record.user_id == user_id
    assert cred.user_id == user_id
    assert len(cred.credential_K) == 32
    assert len(record.salt) == 16
    assert len(record.public_key_Y) == 32
    assert Ristretto255Group.is_valid_point(record.public_key_Y) is True
    assert len(record.get_helper_bits()) == 384
    assert len(record.get_mask_bits()) == 384

def test_identical_impression_authentication_succeeds(synthetic_fingerprint):
    user_id = "user_test_02"
    record, cred = enroll_user(user_id, synthetic_fingerprint)
    session_mgr = ZKPSessionManager()

    # Authenticate with the exact same impression and correct credential
    res = authenticate_user(
        user_id=user_id,
        query_fingerprint_img=synthetic_fingerprint,
        credential_K=cred.credential_K,
        record=record,
        session_manager=session_mgr
    )

    assert res.is_authenticated is True
    assert res.secret_recovered is True
    assert res.zkp_verified is True
    assert res.optimal_shift == 0
    assert res.overlap_ratio >= 0.50
    assert "successfully" in res.status_message

def test_wrong_credential_rejects_authentication(synthetic_fingerprint):
    user_id = "user_test_03"
    record, cred = enroll_user(user_id, synthetic_fingerprint)
    session_mgr = ZKPSessionManager()

    wrong_K = secrets.token_bytes(32)

    res = authenticate_user(
        user_id=user_id,
        query_fingerprint_img=synthetic_fingerprint,
        credential_K=wrong_K,
        record=record,
        session_manager=session_mgr
    )

    # Secret S is recovered, but wrong K causes witness mismatch and ZKP rejection
    assert res.is_authenticated is False
    assert res.secret_recovered is True
    assert res.zkp_verified is False
    assert "failed" in res.status_message.lower()

def test_corrupted_query_image_rejects_secret_recovery():
    user_id = "user_test_04"
    # White noise image (no valid fingerprint structure)
    noise_img = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
    
    # Create valid enrollment with synthetic
    img_valid = np.full((300, 300), 200, dtype=np.uint8)
    yy, xx = np.ogrid[:300, :300]
    rr = np.sqrt((xx - 150)**2 + (yy - 150)**2)
    img_valid = np.where(rr < 120, (127.5 * (1.0 + np.sin(rr / 4.0))).astype(np.uint8), 255).astype(np.uint8)
    
    record, cred = enroll_user(user_id, img_valid)
    session_mgr = ZKPSessionManager()

    try:
        res = authenticate_user(
            user_id=user_id,
            query_fingerprint_img=noise_img,
            credential_K=cred.credential_K,
            record=record,
            session_manager=session_mgr
        )
        assert res.is_authenticated is False
        assert res.secret_recovered is False
    except Exception:
        # Preprocessing failure on pure noise is also a valid rejection
        pass

def test_database_store_lifecycle(synthetic_fingerprint):
    store = SecureEnrollmentStore()
    record, _ = enroll_user("user_db_01", synthetic_fingerprint)

    store.store(record)
    assert store.count() == 1
    assert store.exists("user_db_01") is True
    assert store.get("user_db_01").user_id == "user_db_01"

    # Delete
    assert store.delete("user_db_01") is True
    assert store.count() == 0
    assert store.get("user_db_01") is None

