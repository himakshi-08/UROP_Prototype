"""
Fuzzy Commitment Template Protection Scheme.
Binds a cryptographically secure random secret to biometric binary template B.
Helper Data W = B XOR Codeword(C)
Commitment H = SHA256(C)
Recovery: C' = B' XOR W = Codeword(C) XOR (B XOR B')
ECC Decoder recovers C if Hamming_Distance(B, B') is within tiled majority-vote capacity.
"""

import hashlib

import numpy as np

from features.template import BITS_PER_SECTOR, N_SECTORS, iter_binary_alignments

from .error_correction import BlockECC
from .key_generation import generate_secret_key


class FuzzyCommitment:
    def __init__(self, secret_bits=64, vector_bits=256):
        self.secret_bits = secret_bits
        self.vector_bits = vector_bits
        self.ecc = BlockECC(message_bits=secret_bits, codeword_bits=vector_bits)

    @staticmethod
    def hash_secret(secret_vector):
        secret_bytes = np.packbits(secret_vector).tobytes()
        return hashlib.sha256(secret_bytes).hexdigest()

    def enroll(self, biometric_vector):
        biometric_vector = np.array(biometric_vector, dtype=np.uint8)
        if len(biometric_vector) != self.vector_bits:
            raise ValueError(f"Expected biometric vector length {self.vector_bits}, got {len(biometric_vector)}")

        _, secret = generate_secret_key(bit_length=self.secret_bits)
        codeword = self.ecc.encode(secret)
        helper_data = np.bitwise_xor(biometric_vector, codeword)
        commitment = self.hash_secret(secret)

        return {
            "helper_data": helper_data,
            "commitment": commitment,
            "secret": secret,
        }

    def recover_secret(self, query_biometric_vector, helper_data, stored_commitment):
        """
        Recover C from B' and helper W. Tries cyclic FingerCode sector rotations
        so a residual in-plane rotation does not break a genuine match. A candidate
        is accepted only if SHA-256(C') matches the stored commitment — impostors
        that fall outside ECC capacity cannot forge H.
        """
        query_biometric_vector = np.array(query_biometric_vector, dtype=np.uint8)
        helper_data = np.array(helper_data, dtype=np.uint8)

        best = None
        for aligned in iter_binary_alignments(query_biometric_vector, N_SECTORS, BITS_PER_SECTOR):
            noisy_codeword = np.bitwise_xor(aligned, helper_data)
            recovered, corrected_errors, _ = self.ecc.decode(noisy_codeword)
            candidate_hash = self.hash_secret(recovered)
            reconstructed = self.ecc.encode(recovered)
            residual = int(np.sum(noisy_codeword != reconstructed))
            result = {
                "success": candidate_hash == stored_commitment,
                "recovered_secret": recovered if candidate_hash == stored_commitment else None,
                "errors_corrected": corrected_errors,
                "residual_errors": residual,
                "candidate_hash": candidate_hash,
            }
            if result["success"]:
                return result
            if best is None or residual < best["residual_errors"]:
                best = result
        return best
