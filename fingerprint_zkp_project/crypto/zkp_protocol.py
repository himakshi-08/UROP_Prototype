"""
Zero-Knowledge Proof Protocol & Session Management Module.
Implements secure interactive session management for the Schnorr Sigma Protocol:
1. Fresh, unpredictable challenge generation.
2. Cryptographic binding of session ID, public key, and commitment.
3. Strict session expiration (configurable TTL).
4. Replay protection (single-use sessions and commitment nonce deduplication cache).
"""

import time
import secrets
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass

from .schnorr import SchnorrVerifier, Ristretto255Group, POINT_BYTES, SCALAR_BYTES

@dataclass
class ZKPSession:
    session_id: str
    user_id: str
    public_key_Y: bytes
    commitment_T: bytes
    challenge_c: bytes
    created_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class ZKPSessionManager:
    """
    Server-side session coordinator with anti-replay and expiration mechanisms.
    """
    def __init__(self, default_ttl: float = 60.0):
        self.default_ttl = default_ttl
        self._active_sessions: Dict[str, ZKPSession] = {}
        # Tracks commitments to prevent nonce reuse attacks across sessions
        self._used_commitments: Set[bytes] = set()

    def _purge_expired(self):
        """Removes expired sessions from memory."""
        now = time.time()
        expired = [sid for sid, s in self._active_sessions.items() if (now - s.created_at) > s.ttl_seconds]
        for sid in expired:
            del self._active_sessions[sid]

    def create_session(
        self,
        user_id: str,
        public_key_Y: bytes,
        commitment_T: bytes,
        ttl_seconds: Optional[float] = None
    ) -> Tuple[str, bytes]:
        """
        Initiates a proof session (Pass 1 & 2).
        
        Args:
            user_id: Identity of the claiming user.
            public_key_Y: Enrolled Ristretto255 public key (32 bytes).
            commitment_T: Prover's ephemeral commitment point T (32 bytes).
            ttl_seconds: Session validity duration.
            
        Returns:
            Tuple of (session_id: str, challenge_c: bytes).
        """
        self._purge_expired()

        if not Ristretto255Group.is_valid_point(public_key_Y):
            raise ValueError("Public key Y is not a valid Ristretto255 group element.")
        if not Ristretto255Group.is_valid_point(commitment_T):
            raise ValueError("Commitment point T is not a valid Ristretto255 group element.")

        # Reused nonce detection
        if commitment_T in self._used_commitments:
            raise ValueError("Nonce reuse detected: commitment point has already been used.")
        self._used_commitments.add(commitment_T)

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        session_id = secrets.token_hex(16)
        challenge_c = SchnorrVerifier.generate_challenge()

        session = ZKPSession(
            session_id=session_id,
            user_id=user_id,
            public_key_Y=public_key_Y,
            commitment_T=commitment_T,
            challenge_c=challenge_c,
            created_at=time.time(),
            ttl_seconds=ttl
        )
        self._active_sessions[session_id] = session

        return session_id, challenge_c

    def verify_response(self, session_id: str, response_z: bytes) -> Tuple[bool, str]:
        """
        Finalizes authentication session and verifies Schnorr response (Pass 3 & 4).
        Enforces single-use session consumption (anti-replay).
        """
        self._purge_expired()

        session = self._active_sessions.pop(session_id, None)
        if session is None:
            return False, "Invalid, already consumed, or expired session ID (replay prevented)."

        if session.is_expired:
            return False, "Session expired."

        if len(response_z) != SCALAR_BYTES:
            return False, "Invalid response scalar length."

        verifier = SchnorrVerifier(session.public_key_Y)
        is_valid = verifier.verify(session.commitment_T, session.challenge_c, response_z)

        if is_valid:
            return True, "Zero-knowledge proof verified successfully."
        else:
            return False, "Mathematical verification of proof equation failed."

