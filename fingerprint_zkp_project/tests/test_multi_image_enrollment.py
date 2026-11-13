"""
Unit and Integration Tests for 5-Image Fingerprint Enrollment and Template Fusion.
Verifies rotational alignment, reliability weighting, mask consensus,
and end-to-end authentication of single query images against fused records.
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
    enroll_user_multi,
    align_and_fuse_5_impressions,
    extract_384_template,
    authenticate_user,
    SecureEnrollmentStore,
    EnrollmentRecord,
    ClientCredential
)

@pytest.fixture
def synthetic_5_impressions():
    """Generates 5 realistic synthetic fingerprint impressions with minor mutual rotations and translations."""
    impressions = []
    # Base ridge pattern
    yy, xx = np.ogrid[:300, :300]
    
    # 5 slightly perturbed impressions (core at 150, 150 +/- small offsets and angles)
    offsets = [(0, 0), (2, -2), (-2, 3), (1, 2), (-1, -1)]
    for dx, dy in offsets:
        img = np.full((300, 300), 200, dtype=np.uint8)
        rr = np.sqrt((xx - (150 + dx))**2 + (yy - (150 + dy))**2)
        ridges = (127.5 * (1.0 + np.sin(rr / 4.0))).astype(np.uint8)
        mask = (rr < 120).astype(np.uint8)
        fp = np.where(mask, ridges, 255).astype(np.uint8)
        impressions.append(fp)
    return impressions

def test_align_and_fuse_5_impressions_validity(synthetic_5_impressions):
    t_fused, m_fused, meta = align_and_fuse_5_impressions(synthetic_5_impressions)
    
    assert len(t_fused) == 384
    assert len(m_fused) == 384
    assert t_fused.dtype == np.uint8
    assert m_fused.dtype == np.uint8
    assert np.all((t_fused == 0) | (t_fused == 1))
    assert np.all((m_fused == 0) | (m_fused == 1))
    assert meta["fused_coverage"] >= 0.50
    assert len(meta["align_shifts"]) == 5
    assert len(meta["qualities"]) == 5

def test_enroll_user_multi_creates_valid_record(synthetic_5_impressions):
    user_id = "user_multi_01"
    record, cred, meta = enroll_user_multi(user_id, synthetic_5_impressions)
    
    assert record.user_id == user_id
    assert cred.user_id == user_id
    assert len(cred.credential_K) == 32
    assert len(record.salt) == 16
    assert len(record.public_key_Y) == 32
    assert Ristretto255Group.is_valid_point(record.public_key_Y) is True
    assert len(record.get_helper_bits()) == 384
    assert len(record.get_mask_bits()) == 384
    assert meta["fused_coverage"] >= 0.50

def test_single_query_authenticates_against_fused_record(synthetic_5_impressions):
    user_id = "user_multi_02"
    record, cred, _ = enroll_user_multi(user_id, synthetic_5_impressions)
    session_mgr = ZKPSessionManager()
    
    # Query with a single impression (e.g. impression 0)
    query_img = synthetic_5_impressions[0]
    res = authenticate_user(
        user_id=user_id,
        query_fingerprint_img=query_img,
        credential_K=cred.credential_K,
        record=record,
        session_manager=session_mgr
    )
    
    assert res.is_authenticated is True
    assert res.secret_recovered is True
    assert res.zkp_verified is True
    assert res.overlap_ratio >= 0.50
    assert "successfully" in res.status_message

def test_multi_image_enrollment_rejects_wrong_credential(synthetic_5_impressions):
    user_id = "user_multi_03"
    record, cred, _ = enroll_user_multi(user_id, synthetic_5_impressions)
    session_mgr = ZKPSessionManager()
    
    wrong_K = secrets.token_bytes(32)
    query_img = synthetic_5_impressions[2]
    
    res = authenticate_user(
        user_id=user_id,
        query_fingerprint_img=query_img,
        credential_K=wrong_K,
        record=record,
        session_manager=session_mgr
    )
    
    assert res.is_authenticated is False
    assert res.secret_recovered is True
    assert res.zkp_verified is False
    assert "failed" in res.status_message.lower()

def test_multi_image_enrollment_rejects_wrong_count(synthetic_5_impressions):
    with pytest.raises(ValueError) as excinfo:
        enroll_user_multi("user_bad_count", synthetic_5_impressions[:4])
    assert "exactly 5 impressions" in str(excinfo.value)

def test_enroll_user_backward_compatibility(synthetic_5_impressions):
    # Single image enrollment
    rec1, cred1 = enroll_user("user_single_compat", synthetic_5_impressions[0])
    assert len(rec1.get_helper_bits()) == 384
    
    # 5-image list enrollment
    rec5, cred5 = enroll_user("user_5_compat", synthetic_5_impressions)
    assert len(rec5.get_helper_bits()) == 384

