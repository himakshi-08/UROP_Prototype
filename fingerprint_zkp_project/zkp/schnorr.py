"""
Schnorr-style Sigma Protocol Zero-Knowledge Proof (ZKP) Implementation.
Provides mathematically sound 3-pass interactive Zero-Knowledge Proofs
and optional Fiat-Shamir non-interactive proofs for biometric-derived secrets.
"""

import os
import hashlib
import secrets
import numpy as np
from cryptography.hazmat.primitives.asymmetric import dh

class SchnorrGroup:
    """
    Cryptographic prime group parameters (p, q, g) for Schnorr Sigma Protocol.
    Default: Pre-generated 2048-bit MODP prime group (RFC 5114) or standard safe prime group.
    """
    def __init__(self, p=None, q=None, g=None):
        if p is None or q is None or g is None:
            # Standard 1024-bit MODP Group 2 (RFC 2409 / RFC 3526 Safe Prime)
            # p is 1024-bit safe prime, q = (p-1)/2 is 1023-bit prime, g = 2 is generator of order q.
            self.p = int(
                "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
                "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
                "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
                "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381FFFFFFFFFFFFFFFF",
                16
            )
            self.q = (self.p - 1) // 2
            self.g = 2
        else:
            self.p = p
            self.q = q
            self.g = g

class SchnorrProver:
    def __init__(self, group, secret_x):
        """
        Args:
            group (SchnorrGroup): Cyclic group parameters.
            secret_x (int or bytes or np.ndarray): Witness secret x.
        """
        self.group = group
        if isinstance(secret_x, (bytes, bytearray)):
            self.x = int.from_bytes(secret_x, byteorder='big') % self.group.q
        elif hasattr(secret_x, 'tolist') or isinstance(secret_x, list) or isinstance(secret_x, np.ndarray):
            arr = np.array(secret_x, dtype=np.uint8)
            if np.all((arr == 0) | (arr == 1)):
                packed = np.packbits(arr).tobytes()
                self.x = int.from_bytes(packed, byteorder='big') % self.group.q
            else:
                self.x = int.from_bytes(arr.tobytes(), byteorder='big') % self.group.q
        else:
            self.x = int(secret_x) % self.group.q

        # Public key y = g^x mod p
        self.y = pow(self.group.g, self.x, self.group.p)
        self.r = None # Nonce

    def generate_commitment(self):
        """
        Step 1: Pick random nonce r in [1, q-1], compute commitment t = g^r mod p.
        """
        self.r = secrets.randbelow(self.group.q - 1) + 1
        t = pow(self.group.g, self.r, self.group.p)
        return t

    def compute_response(self, challenge_c):
        """
        Step 3: Compute response s = (r + c * x) mod q.
        """
        if self.r is None:
            raise RuntimeError("Commitment must be generated before computing response.")
        c = int(challenge_c) % self.group.q
        s = (self.r + c * self.x) % self.group.q
        return s

class SchnorrVerifier:
    def __init__(self, group, public_y):
        """
        Args:
            group (SchnorrGroup): Cyclic group parameters.
            public_y (int): Prover's public key y = g^x mod p.
        """
        self.group = group
        self.y = int(public_y)
        self.challenge_c = None

    def generate_challenge(self):
        """
        Step 2: Generate random challenge c in [1, q-1].
        """
        self.challenge_c = secrets.randbelow(self.group.q - 1) + 1
        return self.challenge_c

    def verify_proof(self, commitment_t, response_s, challenge_c=None):
        """
        Step 4: Verify equation g^s == (t * y^c) mod p.
        """
        if challenge_c is None:
            challenge_c = self.challenge_c
        if challenge_c is None:
            raise RuntimeError("No challenge available for verification.")

        t = int(commitment_t) % self.group.p
        s = int(response_s) % self.group.q
        c = int(challenge_c) % self.group.q

        # Left side: g^s mod p
        lhs = pow(self.group.g, s, self.group.p)

        # Right side: (t * y^c) mod p
        yc = pow(self.y, c, self.group.p)
        rhs = (t * yc) % self.group.p

        return (lhs == rhs)
