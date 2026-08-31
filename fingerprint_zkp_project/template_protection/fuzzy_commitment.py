"""
Fuzzy Commitment Template Protection Scheme.
Binds a cryptographically secure random secret to biometric binary template B.
Helper Data W = B XOR Codeword(C)
Commitment H = SHA256(C)
Recovery: C' = B' XOR W = Codeword(C) XOR (B XOR B')
ECC Decoder recovers C if Hamming_Distance(B, B') <= ECC_Capacity.
"""

import hashlib
import numpy as np
from .key_generation import generate_secret_key
from .error_correction import BlockECC

class FuzzyCommitment:
    def __init__(self, secret_bits=64, vector_bits=256):
        self.secret_bits = secret_bits
        self.vector_bits = vector_bits
        self.ecc = BlockECC(message_bits=secret_bits, codeword_bits=vector_bits)

    @staticmethod
    def hash_secret(secret_vector):
        """
        Computes SHA-256 hash of secret binary vector.
        """
        secret_bytes = np.packbits(secret_vector).tobytes()
        return hashlib.sha256(secret_bytes).hexdigest()

    def enroll(self, biometric_vector):
        """
        Enrolls a biometric vector into the Fuzzy Commitment scheme.
        
        Args:
            biometric_vector (np.ndarray): Binary biometric representation B of length `vector_bits`.
            
        Returns:
            dict containing:
                - helper_data (np.ndarray): Protected helper data W = B XOR C_enc
                - commitment (str): Cryptographic hash H = SHA256(C)
                - secret (np.ndarray): Retained during enrollment phase for local verification testing
        """
        biometric_vector = np.array(biometric_vector, dtype=np.uint8)
        if len(biometric_vector) != self.vector_bits:
            raise ValueError(f"Expected biometric vector length {self.vector_bits}, got {len(biometric_vector)}")

        # Step 1: Generate random cryptographic secret C
        _, secret = generate_secret_key(bit_length=self.secret_bits)

        # Step 2: Encode secret C using ECC to get codeword C_enc
        codeword = self.ecc.encode(secret)

        # Step 3: Compute protected helper data W = B XOR C_enc
        helper_data = np.bitwise_xor(biometric_vector, codeword)

        # Step 4: Compute cryptographic commitment H = SHA256(C)
        commitment = self.hash_secret(secret)

        return {
            "helper_data": helper_data,
            "commitment": commitment,
            "secret": secret
        }

    def recover_secret(self, query_biometric_vector, helper_data, stored_commitment):
        """
        Attempts to recover secret key C using query biometric vector B' and stored helper data W.
        
        Returns:
            dict containing:
                - success (bool): True if SHA256(recovered_C) == stored_commitment
                - recovered_secret (np.ndarray or None)
                - errors_corrected (int)
                - bit_difference (int): Hamming distance between B and B'
        """
        query_biometric_vector = np.array(query_biometric_vector, dtype=np.uint8)
        helper_data = np.array(helper_data, dtype=np.uint8)

        # Step 1: Compute noisy codeword C' = B' XOR W
        noisy_codeword = np.bitwise_xor(query_biometric_vector, helper_data)

        # Step 2: Apply ECC decoder to recover candidate secret
        recovered_secret, corrected_errors, _ = self.ecc.decode(noisy_codeword)

        # Step 3: Verify candidate secret against stored cryptographic commitment hash
        candidate_hash = self.hash_secret(recovered_secret)
        is_valid = (candidate_hash == stored_commitment)

        return {
            "success": is_valid,
            "recovered_secret": recovered_secret if is_valid else None,
            "errors_corrected": corrected_errors,
            "candidate_hash": candidate_hash
        }
