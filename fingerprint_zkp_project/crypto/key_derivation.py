"""
Cryptographic Key Derivation Module.
Derives the Schnorr private witness scalar x from the joint combination of:
1. S: Biometric secret recovered from the Fuzzy Commitment Scheme (32 bits).
2. K: Independent high-entropy user credential (256 bits).

Uses HKDF-SHA256 (RFC 5869) with context separation, then applies Ristretto255
scalar reduction (RFC 9496) to produce a uniform, canonical scalar modulo L.
"""

import os
from typing import Union
import numpy as np
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

from .schnorr import Ristretto255Group, SCALAR_BYTES

DEFAULT_CONTEXT = b"Fingerprint-ZKP-Schnorr-Witness-v1"

def _normalize_secret_bytes(secret: Union[bytes, bytearray, np.ndarray, list]) -> bytes:
    """Normalizes a 32-bit biometric secret into 4 bytes."""
    if isinstance(secret, (bytes, bytearray)):
        return bytes(secret)
    elif isinstance(secret, (list, np.ndarray)):
        arr = np.array(secret, dtype=np.uint8)
        if len(arr) == 32 and np.all((arr == 0) | (arr == 1)):
            return np.packbits(arr).tobytes()
        elif len(arr) == 4:
            return arr.tobytes()
        else:
            return np.packbits(arr).tobytes()
    else:
        raise TypeError(f"Unsupported secret type: {type(secret)}")

def derive_witness(
    biometric_secret: Union[bytes, bytearray, np.ndarray, list],
    user_credential: bytes,
    salt: bytes,
    context: bytes = DEFAULT_CONTEXT
) -> bytes:
    """
    Derives the Schnorr private witness scalar x.
    
    Args:
        biometric_secret: 32-bit secret recovered by ECC from the fingerprint.
        user_credential: High-entropy independent credential K (>= 16 bytes, preferably 32 bytes).
        salt: Unique per-user random salt (>= 16 bytes).
        context: Domain separation string.
        
    Returns:
        32-byte canonical Ristretto255 scalar x.
    """
    s_bytes = _normalize_secret_bytes(biometric_secret)
    if not isinstance(user_credential, (bytes, bytearray)) or len(user_credential) < 16:
        raise ValueError("User credential K must be at least 16 bytes (256 bits recommended).")
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < 16:
        raise ValueError("Salt must be at least 16 bytes.")

    ikm = s_bytes + bytes(user_credential)

    # 1. HKDF-Extract and Expand to 64 bytes using SHA-256
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=salt,
        info=context
    )
    raw_64 = hkdf.derive(ikm)

    # 2. Wide reduction modulo L (Ristretto255 group order)
    x = Ristretto255Group.reduce_scalar_64(raw_64)
    return x

