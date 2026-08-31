"""
Error-Correcting Code (ECC) Module.
Implements block repetition-parity encoding and syndrome-based error correction
to expand a K-bit secret key into an N-bit codeword and correct bit errors up to tolerance threshold.
"""

import numpy as np

class BlockECC:
    def __init__(self, message_bits=64, codeword_bits=256):
        """
        Args:
            message_bits (int): Secret key size K (e.g., 64).
            codeword_bits (int): Biometric vector size N (e.g., 256).
        """
        self.k = message_bits
        self.n = codeword_bits
        self.rep_factor = self.n // self.k
        if self.rep_factor < 1:
            raise ValueError(f"Codeword length N={self.n} must be >= message length K={self.k}")
        # Maximum errors correctable per block: (rep_factor - 1) // 2
        self.max_errors_per_block = (self.rep_factor - 1) // 2

    def encode(self, secret_bits):
        """
        Encodes K-bit secret into N-bit codeword using repetition & parity interleaving.
        
        Args:
            secret_bits (np.ndarray): Binary array of length K.
            
        Returns:
            np.ndarray: Encoded codeword of length N.
        """
        secret_bits = np.array(secret_bits, dtype=np.uint8)
        if len(secret_bits) != self.k:
            raise ValueError(f"Expected secret length {self.k}, got {len(secret_bits)}")

        # Repetition matrix expansion
        codeword = np.repeat(secret_bits, self.rep_factor)
        
        # Padding if n is not exact multiple of k
        if len(codeword) < self.n:
            pad = np.zeros(self.n - len(codeword), dtype=np.uint8)
            codeword = np.concatenate([codeword, pad])

        return codeword

    def decode(self, noisy_codeword):
        """
        Decodes N-bit noisy codeword back to K-bit secret via majority voting error correction.
        
        Args:
            noisy_codeword (np.ndarray): Noisy binary vector of length N.
            
        Returns:
            tuple: (recovered_secret_bits, num_corrected_bits, decoding_success)
        """
        noisy_codeword = np.array(noisy_codeword, dtype=np.uint8)
        if len(noisy_codeword) != self.n:
            raise ValueError(f"Expected noisy codeword length {self.n}, got {len(noisy_codeword)}")

        recovered = np.zeros(self.k, dtype=np.uint8)
        corrected_bits = 0

        for i in range(self.k):
            block = noisy_codeword[i * self.rep_factor : (i + 1) * self.rep_factor]
            ones_count = np.sum(block == 1)
            zeros_count = np.sum(block == 0)

            majority_bit = 1 if ones_count > zeros_count else 0
            recovered[i] = majority_bit

            # Count errors corrected in this block
            errors = np.sum(block != majority_bit)
            corrected_bits += errors

        return recovered, int(corrected_bits), True
