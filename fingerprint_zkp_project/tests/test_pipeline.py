"""
Unit tests for End-to-End Integrated Biometric Authentication Pipeline.
"""

import numpy as np

from authentication import SecureBiometricPipeline

def test_full_pipeline_enrollment_and_authentication():
    pipeline = SecureBiometricPipeline(secret_bits=64, vector_bits=256)
    
    # Generate synthetic image sample for user 101
    img_enroll = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    img_query = img_enroll.copy()

    # Add minor noise
    noise = np.random.normal(0, 5, (150, 150)).astype(np.int16)
    img_query = np.clip(img_query.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. Enroll User 101
    enroll_res = pipeline.enroll("user_101", img_enroll)
    assert enroll_res["status"] == "ENROLLED"

    # Verify user record is in database
    db_record = pipeline.db.get_user("user_101")
    assert db_record is not None
    assert "helper_data" in db_record
    assert "commitment" in db_record
    assert "public_key" in db_record

    # 2. Authenticate User 101 with query image
    auth_res = pipeline.authenticate("user_101", img_query)
    assert "authenticated" in auth_res
