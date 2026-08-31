"""
Baseline Fingerprint Matcher.
Determines whether two fingerprint templates belong to the same finger
by computing minutiae similarity scores and binary template similarity scores.
"""

from .distance import match_minutiae_sets
from features.template import BiometricTemplate

class FingerprintMatcher:
    def __init__(self, default_threshold=0.25, mode="combined"):
        """
        Args:
            default_threshold (float): Similarity threshold for decision.
            mode (str): Matching mode - "minutiae", "binary", or "combined".
        """
        self.threshold = default_threshold
        self.mode = mode

    def match(self, template_a, template_b, threshold=None):
        """
        Matches template_a against template_b.
        
        Returns:
            dict containing:
                - score (float)
                - is_match (bool)
                - threshold (float)
        """
        if threshold is None:
            threshold = self.threshold

        if self.mode == "minutiae":
            score = match_minutiae_sets(template_a.minutiae, template_b.minutiae)
        elif self.mode == "binary":
            score = BiometricTemplate.hamming_similarity(template_a.binary_vector, template_b.binary_vector)
        else: # "combined"
            score_m = match_minutiae_sets(template_a.minutiae, template_b.minutiae)
            score_b = BiometricTemplate.hamming_similarity(template_a.binary_vector, template_b.binary_vector)
            score = 0.6 * score_m + 0.4 * score_b

        return {
            "score": float(score),
            "is_match": bool(score >= threshold),
            "threshold": float(threshold)
        }
