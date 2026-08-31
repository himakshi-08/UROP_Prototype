"""
End-to-End Privacy-Preserving Biometric Authentication Pipeline.
Integrates Biometric Processing, Fuzzy Commitment Template Protection, and Schnorr ZKP.
"""

import time
import numpy as np
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, BiometricTemplate
from template_protection import FuzzyCommitment
from zkp import SchnorrGroup, UserDeviceProver, ServerVerifier

class AuthenticationDatabase:
    """
    Simulated Secure Authentication Database.
    Stores ONLY protected helper data W, cryptographic commitment H, and public key Y.
    NEVER stores raw fingerprints or original biometric templates.
    """
    def __init__(self):
        self.records = {}

    def store_enrollment(self, user_id, helper_data, commitment, public_key):
        self.records[user_id] = {
            "helper_data": helper_data,
            "commitment": commitment,
            "public_key": public_key
        }

    def get_user(self, user_id):
        return self.records.get(user_id, None)

class SecureBiometricPipeline:
    def __init__(self, secret_bits=64, vector_bits=256):
        self.secret_bits = secret_bits
        self.vector_bits = vector_bits
        self.fc = FuzzyCommitment(secret_bits=secret_bits, vector_bits=vector_bits)
        self.group = SchnorrGroup()
        self.db = AuthenticationDatabase()

    def enroll(self, user_id, fingerprint_img):
        """
        Executes Enrollment Workflow.
        """
        # Step 1: Preprocessing & Minutiae Feature Extraction
        prep = preprocess_fingerprint(fingerprint_img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        template = BiometricTemplate(minutiae, image_shape=fingerprint_img.shape, vector_bits=self.vector_bits)

        # Step 2: Biometric Template Protection via Fuzzy Commitment
        enroll_res = self.fc.enroll(template.binary_vector)
        helper_data = enroll_res["helper_data"]
        commitment = enroll_res["commitment"]
        secret = enroll_res["secret"]

        # Step 3: Compute Schnorr ZKP Public Key Y = g^secret mod p
        prover_temp = UserDeviceProver(secret, group=self.group)
        public_key = prover_temp.public_key

        # Step 4: Store only protected reference in Database
        self.db.store_enrollment(user_id, helper_data, commitment, public_key)

        return {
            "user_id": user_id,
            "status": "ENROLLED",
            "num_minutiae": len(minutiae),
            "commitment": commitment
        }

    def authenticate(self, user_id, query_fingerprint_img):
        """
        Executes Authentication Workflow with Zero-Knowledge Proof.
        
        Returns:
            dict containing timing, secret recovery status, ZKP status, and final Accept/Reject result.
        """
        start_time = time.perf_counter()

        record = self.db.get_user(user_id)
        if record is None:
            return {"authenticated": False, "reason": "User not found"}

        helper_data = record["helper_data"]
        stored_commitment = record["commitment"]
        public_key = record["public_key"]

        # Step 1: Preprocess query image & extract binary template B'
        t0 = time.perf_counter()
        prep = preprocess_fingerprint(query_fingerprint_img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        query_template = BiometricTemplate(minutiae, image_shape=query_fingerprint_img.shape, vector_bits=self.vector_bits)
        t_biometric = time.perf_counter() - t0

        # Step 2: Attempt Fuzzy Commitment Secret Recovery
        t0 = time.perf_counter()
        fc_res = self.fc.recover_secret(query_template.binary_vector, helper_data, stored_commitment)
        t_fc = time.perf_counter() - t0

        if not fc_res["success"]:
            total_time = time.perf_counter() - start_time
            return {
                "authenticated": False,
                "reason": "Biometric recovery failed (too many bit errors)",
                "errors_corrected": fc_res["errors_corrected"],
                "total_time_ms": total_time * 1000
            }

        recovered_secret = fc_res["recovered_secret"]

        # Step 3: Zero-Knowledge Proof Authentication Exchange
        t0 = time.perf_counter()
        prover = UserDeviceProver(recovered_secret, group=self.group)
        t_proof_gen_start = time.perf_counter()
        commitment_t = prover.step1_create_commitment()
        t_proof_gen = time.perf_counter() - t_proof_gen_start

        verifier = ServerVerifier(public_key, group=self.group)
        challenge_c = verifier.step1_issue_challenge()

        t_response_start = time.perf_counter()
        response_s = prover.step2_respond_to_challenge(challenge_c)
        t_proof_gen += (time.perf_counter() - t_response_start)

        t_verify_start = time.perf_counter()
        zkp_valid = verifier.step2_verify(commitment_t, response_s, challenge_c=challenge_c)
        t_proof_verify = time.perf_counter() - t_verify_start

        total_time = time.perf_counter() - start_time

        return {
            "authenticated": bool(zkp_valid),
            "secret_recovered": True,
            "zkp_verified": bool(zkp_valid),
            "errors_corrected": fc_res["errors_corrected"],
            "timing_ms": {
                "biometric_processing": t_biometric * 1000,
                "fuzzy_commitment_recovery": t_fc * 1000,
                "proof_generation": t_proof_gen * 1000,
                "proof_verification": t_proof_verify * 1000,
                "total": total_time * 1000
            }
        }
