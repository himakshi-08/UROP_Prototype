"""
Error-Correcting Code (ECC) Module.
Tiled block-repetition encoding with a public interleaver, decoded by majority vote.

Tiled replicas let a spatial burst in B hit only one copy of the secret.
A fixed public permutation then spreads residual periodic FingerCode errors
so they do not land on the same secret bit in every replica.
Each secret bit survives floor((R-1)/2) replica errors.
"""

import numpy as np


def _public_interleaver(n):
    rng = np.random.RandomState(20260921)
    perm = rng.permutation(n).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    return perm, inv


class BlockECC:
    def __init__(self, message_bits=64, codeword_bits=256):
        self.k = message_bits
        self.n = codeword_bits
        self.rep_factor = self.n // self.k
        if self.rep_factor < 1:
            raise ValueError(f"Codeword length N={self.n} must be >= message length K={self.k}")
        self.max_errors_per_block = (self.rep_factor - 1) // 2
        self.perm, self.inv = _public_interleaver(self.n)

    def encode(self, secret_bits):
        secret_bits = np.array(secret_bits, dtype=np.uint8)
        if len(secret_bits) != self.k:
            raise ValueError(f"Expected secret length {self.k}, got {len(secret_bits)}")

        replicas = np.tile(secret_bits, self.rep_factor)
        if len(replicas) < self.n:
            pad = np.zeros(self.n - len(replicas), dtype=np.uint8)
            replicas = np.concatenate([replicas, pad])
        replicas = replicas[: self.n]
        return replicas[self.perm]

    def decode(self, noisy_codeword):
        noisy_codeword = np.array(noisy_codeword, dtype=np.uint8)
        if len(noisy_codeword) != self.n:
            raise ValueError(f"Expected noisy codeword length {self.n}, got {len(noisy_codeword)}")

        deint = noisy_codeword[self.inv]
        usable = self.rep_factor * self.k
        replicas = deint[:usable].reshape(self.rep_factor, self.k)
        ones_count = np.sum(replicas, axis=0)
        recovered = (ones_count > (self.rep_factor / 2.0)).astype(np.uint8)

        reconstructed = self.encode(recovered)
        corrected_bits = int(np.sum(noisy_codeword != reconstructed))
        return recovered, corrected_bits, True
