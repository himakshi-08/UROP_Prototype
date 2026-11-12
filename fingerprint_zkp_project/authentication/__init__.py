"""
Authentication Package.
Integrates 384-bit feature extraction, Fuzzy Commitment Scheme with interleaved repetition,
and Ristretto255 Schnorr Zero-Knowledge Proof authentication.
"""

from .database import EnrollmentRecord, ClientCredential, SecureEnrollmentStore
from .enrollment import enroll_user, extract_384_template, get_public_interleaver
from .authenticate import authenticate_user, AuthenticationResult
from .authentication_pipeline import SecureBiometricPipeline, AuthenticationDatabase

__all__ = [
    "EnrollmentRecord",
    "ClientCredential",
    "SecureEnrollmentStore",
    "enroll_user",
    "extract_384_template",
    "get_public_interleaver",
    "authenticate_user",
    "AuthenticationResult",
    "SecureBiometricPipeline",
    "AuthenticationDatabase"
]
