"""
Authentication Server Verifier Interface.
Runs on server to issue random challenge and verify Zero-Knowledge Proof without knowing secret.
"""

from .schnorr import SchnorrGroup, SchnorrVerifier

class ServerVerifier:
    def __init__(self, public_key_y, group=None):
        if group is None:
            group = SchnorrGroup()
        self.group = group
        self.verifier = SchnorrVerifier(self.group, public_key_y)

    def step1_issue_challenge(self):
        """
        Issues fresh random challenge c.
        """
        return self.verifier.generate_challenge()

    def step2_verify(self, commitment_t, response_s, challenge_c=None):
        """
        Verifies Zero-Knowledge Proof s for commitment t.
        
        Returns:
            bool: True if proof is cryptographically valid, False otherwise.
        """
        return self.verifier.verify_proof(commitment_t, response_s, challenge_c=challenge_c)
