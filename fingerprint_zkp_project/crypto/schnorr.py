"""
Ristretto255 Schnorr Zero-Knowledge Proof (Sigma Protocol) Implementation.
Uses libsodium's native Ristretto255 group operations (RFC 9496) to provide
a canonical prime-order group of order L = 2^252 + 27742317777372353535851937790883648493.

Eliminates the cofactor-8 and small-subgroup vulnerabilities of raw Ed25519,
and replaces legacy 1024-bit MODP parameters with modern 128-bit security.
"""

import os
import sys
import ctypes
import secrets
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Libsodium Dynamic Library Loader
# ---------------------------------------------------------------------------
def _load_libsodium():
    # 1. Check local directory (fingerprint_zkp_project/crypto/libsodium.dll)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    local_candidates = [
        os.path.join(curr_dir, "libsodium.dll"),
        os.path.join(curr_dir, "libsodium.so"),
        os.path.join(curr_dir, "libsodium.dylib"),
        os.path.join(curr_dir, "libsodium.so.23"),
    ]
    for c in local_candidates:
        if os.path.exists(c):
            try:
                lib = ctypes.cdll.LoadLibrary(c)
                if lib.sodium_init() >= 0:
                    return lib
            except Exception:
                pass

    # 2. Check SODIUM_LIBRARY_PATH environment variable
    env_path = os.environ.get("SODIUM_LIBRARY_PATH")
    if env_path and os.path.exists(env_path):
        try:
            lib = ctypes.cdll.LoadLibrary(env_path)
            if lib.sodium_init() >= 0:
                return lib
        except Exception:
            pass

    # 3. System dynamic library resolution
    for name in ["sodium", "libsodium", "libsodium.so.23", "libsodium-23.dll", "libsodium.dll"]:
        try:
            lib = ctypes.cdll.LoadLibrary(name)
            if lib.sodium_init() >= 0:
                return lib
        except Exception:
            pass

    raise RuntimeError(
        "Could not load libsodium dynamic library. Please ensure libsodium.dll / libsodium.so "
        "is present in the crypto/ directory or available on PATH / SODIUM_LIBRARY_PATH."
    )

_sodium = _load_libsodium()

# Point and scalar sizes for Ristretto255 (RFC 9496)
POINT_BYTES = 32
SCALAR_BYTES = 32

# Configure libsodium ctypes signatures
_sodium.crypto_core_ristretto255_is_valid_point.argtypes = [ctypes.c_char_p]
_sodium.crypto_core_ristretto255_is_valid_point.restype = ctypes.c_int

_sodium.crypto_core_ristretto255_scalar_random.argtypes = [ctypes.c_char_p]
_sodium.crypto_core_ristretto255_scalar_random.restype = None

_sodium.crypto_scalarmult_ristretto255_base.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_scalarmult_ristretto255_base.restype = ctypes.c_int

_sodium.crypto_scalarmult_ristretto255.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_scalarmult_ristretto255.restype = ctypes.c_int

_sodium.crypto_core_ristretto255_add.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_add.restype = ctypes.c_int

_sodium.crypto_core_ristretto255_sub.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_sub.restype = ctypes.c_int

_sodium.crypto_core_ristretto255_scalar_add.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_scalar_add.restype = None

_sodium.crypto_core_ristretto255_scalar_sub.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_scalar_sub.restype = None

_sodium.crypto_core_ristretto255_scalar_mul.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_scalar_mul.restype = None

_sodium.crypto_core_ristretto255_scalar_reduce.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_sodium.crypto_core_ristretto255_scalar_reduce.restype = None


# ---------------------------------------------------------------------------
# High-Level Ristretto255 Group Operations
# ---------------------------------------------------------------------------
class Ristretto255Group:
    """Provides pure Ristretto255 group operations."""
    
    @staticmethod
    def is_valid_point(p_bytes: bytes) -> bool:
        if len(p_bytes) != POINT_BYTES:
            return False
        return bool(_sodium.crypto_core_ristretto255_is_valid_point(p_bytes) == 1)

    @staticmethod
    def random_scalar() -> bytes:
        buf = ctypes.create_string_buffer(SCALAR_BYTES)
        _sodium.crypto_core_ristretto255_scalar_random(buf)
        return buf.raw

    @staticmethod
    def reduce_scalar_64(bytes_64: bytes) -> bytes:
        if len(bytes_64) != 64:
            raise ValueError("reduce_scalar_64 requires exactly 64 bytes of input material.")
        buf = ctypes.create_string_buffer(SCALAR_BYTES)
        _sodium.crypto_core_ristretto255_scalar_reduce(buf, bytes_64)
        return buf.raw

    @staticmethod
    def base_scalarmult(scalar_bytes: bytes) -> bytes:
        if len(scalar_bytes) != SCALAR_BYTES:
            raise ValueError(f"Scalar must be {SCALAR_BYTES} bytes.")
        buf = ctypes.create_string_buffer(POINT_BYTES)
        res = _sodium.crypto_scalarmult_ristretto255_base(buf, scalar_bytes)
        if res != 0 or not Ristretto255Group.is_valid_point(buf.raw):
            raise ValueError("Base scalar multiplication produced an invalid group element.")
        return buf.raw

    @staticmethod
    def scalarmult(scalar_bytes: bytes, point_bytes: bytes) -> bytes:
        if len(scalar_bytes) != SCALAR_BYTES or len(point_bytes) != POINT_BYTES:
            raise ValueError(f"Scalar must be {SCALAR_BYTES} bytes, Point must be {POINT_BYTES} bytes.")
        if not Ristretto255Group.is_valid_point(point_bytes):
            raise ValueError("Input point is not a valid Ristretto255 group element.")
        buf = ctypes.create_string_buffer(POINT_BYTES)
        res = _sodium.crypto_scalarmult_ristretto255(buf, scalar_bytes, point_bytes)
        if res != 0 or not Ristretto255Group.is_valid_point(buf.raw):
            raise ValueError("Scalar multiplication produced an invalid group element.")
        return buf.raw

    @staticmethod
    def point_add(p1: bytes, p2: bytes) -> bytes:
        if not Ristretto255Group.is_valid_point(p1) or not Ristretto255Group.is_valid_point(p2):
            raise ValueError("Cannot add invalid Ristretto255 points.")
        buf = ctypes.create_string_buffer(POINT_BYTES)
        res = _sodium.crypto_core_ristretto255_add(buf, p1, p2)
        if res != 0:
            raise ValueError("Point addition failed.")
        return buf.raw

    @staticmethod
    def scalar_mul(s1: bytes, s2: bytes) -> bytes:
        if len(s1) != SCALAR_BYTES or len(s2) != SCALAR_BYTES:
            raise ValueError("Scalars must be 32 bytes.")
        buf = ctypes.create_string_buffer(SCALAR_BYTES)
        _sodium.crypto_core_ristretto255_scalar_mul(buf, s1, s2)
        return buf.raw

    @staticmethod
    def scalar_add(s1: bytes, s2: bytes) -> bytes:
        if len(s1) != SCALAR_BYTES or len(s2) != SCALAR_BYTES:
            raise ValueError("Scalars must be 32 bytes.")
        buf = ctypes.create_string_buffer(SCALAR_BYTES)
        _sodium.crypto_core_ristretto255_scalar_add(buf, s1, s2)
        return buf.raw


# ---------------------------------------------------------------------------
# Schnorr Prover & Verifier Classes
# ---------------------------------------------------------------------------
class SchnorrProver:
    """
    Ristretto255 Schnorr Sigma Protocol Prover.
    Proves knowledge of secret witness scalar x such that Y = x * B.
    """
    def __init__(self, private_scalar_x: bytes):
        if len(private_scalar_x) != SCALAR_BYTES:
            raise ValueError(f"Witness scalar must be exactly {SCALAR_BYTES} bytes.")
        self.x = private_scalar_x
        # Public key Y = x * B
        self.Y = Ristretto255Group.base_scalarmult(self.x)
        self._nonce_r: Optional[bytes] = None

    @property
    def public_key(self) -> bytes:
        return self.Y

    def commit(self) -> bytes:
        """
        Pass 1: Generate fresh ephemeral nonce r, compute commitment T = r * B.
        """
        self._nonce_r = Ristretto255Group.random_scalar()
        T = Ristretto255Group.base_scalarmult(self._nonce_r)
        return T

    def respond(self, challenge_c: bytes) -> bytes:
        """
        Pass 3: Compute response scalar z = (r + c * x) mod L.
        Enforces single-use nonce: self._nonce_r is cleared immediately after use.
        """
        if self._nonce_r is None:
            raise RuntimeError("Nonce has already been used or commitment was not created.")
        if len(challenge_c) != SCALAR_BYTES:
            raise ValueError(f"Challenge must be {SCALAR_BYTES} bytes.")

        # cx = (c * x) mod L
        cx = Ristretto255Group.scalar_mul(challenge_c, self.x)
        # z = (r + cx) mod L
        z = Ristretto255Group.scalar_add(self._nonce_r, cx)
        
        # Zeroize and invalidate nonce to prevent nonce reuse attacks
        self._nonce_r = None
        return z


class SchnorrVerifier:
    """
    Ristretto255 Schnorr Sigma Protocol Verifier.
    Verifies that z * B == T + c * Y without learning x.
    """
    def __init__(self, public_key_Y: bytes):
        if not Ristretto255Group.is_valid_point(public_key_Y):
            raise ValueError("Public key Y is not a valid Ristretto255 group element.")
        self.Y = public_key_Y

    @staticmethod
    def generate_challenge() -> bytes:
        """
        Pass 2: Generate random challenge scalar c in [1, L-1].
        """
        return Ristretto255Group.random_scalar()

    def verify(self, commitment_T: bytes, challenge_c: bytes, response_z: bytes) -> bool:
        """
        Pass 4: Verify the Schnorr relation z * B == T + c * Y.
        """
        # Validate inputs
        if len(challenge_c) != SCALAR_BYTES or len(response_z) != SCALAR_BYTES:
            return False
        if not Ristretto255Group.is_valid_point(commitment_T):
            return False

        try:
            # LHS = z * B
            lhs = Ristretto255Group.base_scalarmult(response_z)
            
            # cY = c * Y
            cY = Ristretto255Group.scalarmult(challenge_c, self.Y)
            
            # RHS = T + c * Y
            rhs = Ristretto255Group.point_add(commitment_T, cY)
            
            # Constant-time comparison
            return secrets.compare_digest(lhs, rhs)
        except Exception:
            return False

