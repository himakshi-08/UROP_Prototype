"""
Secure Enrollment Database Store.
Stores only non-sensitive public and helper artifacts:
- user_id: Unique user identifier
- helper_data: W = T_enroll XOR Codeword(S) (384 bits)
- mask: Cell validity mask M_enroll (384 bits)
- salt: Per-user random salt for HKDF (16 bytes)
- public_key_Y: Ristretto255 public verification key (32 bytes)
- version: Protocol and template version

SECURITY INVARIANTS:
1. NEVER stores the raw biometric template T_enroll.
2. NEVER stores the biometric secret S.
3. NEVER stores the independent user credential K.
4. NEVER stores the private witness scalar x.
5. NEVER stores a weak or unsalted hash of the secret (e.g. SHA256(S)).
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union
import numpy as np

@dataclass
class EnrollmentRecord:
    user_id: str
    helper_data: bytes          # 384 bits packed or bytes
    mask: bytes                 # 384 bits packed or bytes
    salt: bytes                 # 16 bytes
    public_key_Y: bytes         # 32 bytes Ristretto255 point
    version: str = "v2.0-384b-ristretto255"

    def get_helper_bits(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.helper_data, dtype=np.uint8))[:384]

    def get_mask_bits(self) -> np.ndarray:
        return np.unpackbits(np.frombuffer(self.mask, dtype=np.uint8))[:384]


@dataclass
class ClientCredential:
    user_id: str
    credential_K: bytes         # 32 bytes (256 bits) high-entropy possession factor


class SecureEnrollmentStore:
    """
    Secure server-side database for public authentication artifacts.
    """
    def __init__(self):
        self._store: Dict[str, EnrollmentRecord] = {}

    def store(self, record: EnrollmentRecord):
        if not isinstance(record, EnrollmentRecord):
            raise TypeError("Expected EnrollmentRecord instance.")
        if len(record.public_key_Y) != 32:
            raise ValueError("Public key Y must be 32 bytes.")
        if len(record.salt) < 16:
            raise ValueError("Salt must be at least 16 bytes.")
        self._store[record.user_id] = record

    def get(self, user_id: str) -> Optional[EnrollmentRecord]:
        return self._store.get(user_id, None)

    def delete(self, user_id: str) -> bool:
        return self._store.pop(user_id, None) is not None

    def exists(self, user_id: str) -> bool:
        return user_id in self._store

    def count(self) -> int:
        return len(self._store)

