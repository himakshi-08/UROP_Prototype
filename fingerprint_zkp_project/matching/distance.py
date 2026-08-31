"""
Distance & Similarity Computation Utilities.
Calculates minutiae pair distances, angular differences, and binary vector similarity.
"""

import numpy as np

def minutia_distance(m1, m2):
    """
    Euclidean distance between two minutiae points.
    """
    return np.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)

def angular_difference(a1, a2):
    """
    Minimal angular difference between two orientation angles in radians.
    """
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)

def match_minutiae_sets(minutiae1, minutiae2, max_dist=15.0, max_angle=np.pi/6):
    """
    Computes baseline matching score between two minutiae sets with alignment search.
    
    Returns:
        float: Similarity match score in range [0.0, 1.0].
    """
    if not minutiae1 or not minutiae2:
        return 0.0

    best_matched_count = 0

    # Test potential reference minutiae alignments
    for ref1 in minutiae1[:min(10, len(minutiae1))]:
        for ref2 in minutiae2[:min(10, len(minutiae2))]:
            dx = ref2.x - ref1.x
            dy = ref2.y - ref1.y
            dtheta = ref2.orientation - ref1.orientation

            # Count matched minutiae under this alignment translation (dx, dy)
            matched = 0
            used_m2 = set()

            for m1 in minutiae1:
                # Transformed coordinates
                x1_t = m1.x + dx
                y1_t = m1.y + dy

                for j, m2 in enumerate(minutiae2):
                    if j in used_m2:
                        continue
                    dist = np.sqrt((x1_t - m2.x)**2 + (y1_t - m2.y)**2)
                    angle_diff = angular_difference(m1.orientation + dtheta, m2.orientation)

                    if dist <= max_dist and angle_diff <= max_angle and m1.type == m2.type:
                        matched += 1
                        used_m2.add(j)
                        break

            if matched > best_matched_count:
                best_matched_count = matched

    # Normalize score relative to average number of minutiae
    score = (2.0 * best_matched_count) / float(len(minutiae1) + len(minutiae2))
    return float(np.clip(score, 0.0, 1.0))
