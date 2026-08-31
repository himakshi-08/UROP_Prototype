"""
Prover Device Interface.
Runs on user device to perform Zero-Knowledge Proof authentication steps.
"""

from .schnorr import SchnorrGroup, SchnorrProver

class UserDeviceProver:
    def __init__(self, recovered_secret, group=None):
        if group is None:
            group = SchnorrGroup()
        self.group = group
        self.prover = SchnorrProver(self.group, recovered_secret)
        self.public_key = self.prover.y

    def step1_create_commitment(self):
        """
        Generates commitment t to send to Authentication Server.
        """
        return self.prover.generate_commitment()

    def step2_respond_to_challenge(self, challenge_c):
        """
        Computes proof response s for received server challenge c.
        """
        return self.prover.compute_response(challenge_c)
