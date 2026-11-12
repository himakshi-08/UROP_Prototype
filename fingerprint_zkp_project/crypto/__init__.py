"""
Cryptographic Package for Privacy-Preserving Zero-Knowledge Authentication.
Provides:
- Ristretto255 Schnorr Sigma Protocol (Prover, Verifier, Group)
- HKDF-SHA256 Witness Key Derivation
- Session Coordination and Anti-Replay Management
"""

from .schnorr import (
    Ristretto255Group,
    SchnorrProver,
    SchnorrVerifier,
    POINT_BYTES,
    SCALAR_BYTES
)
from .key_derivation import (
    derive_witness,
    DEFAULT_CONTEXT
)
from .zkp_protocol import (
    ZKPSession,
    ZKPSessionManager
)

__all__ = [
    "Ristretto255Group",
    "SchnorrProver",
    "SchnorrVerifier",
    "POINT_BYTES",
    "SCALAR_BYTES",
    "derive_witness",
    "DEFAULT_CONTEXT",
    "ZKPSession",
    "ZKPSessionManager"
]

