"""
Distance & Similarity Computation Utilities.
Calculates minutiae pair distances, angular differences, and binary vector similarity.
"""

import numpy as np


def minutia_distance(m1, m2):
    return np.sqrt((m1.x - m2.x) ** 2 + (m1.y - m2.y) ** 2)


def angular_difference(a1, a2):
    diff = abs(a1 - a2) % np.pi
    return min(diff, np.pi - diff)


def match_minutiae_sets(minutiae1, minutiae2, max_dist=12.0, max_angle=np.pi / 8):
    """
    Rigid alignment search: each pair of reference minutiae proposes (dx, dy, dθ).
    Score is 2 * matches / (n1 + n2).
    """
    if not minutiae1 or not minutiae2:
        return 0.0

    m1_list = list(minutiae1)
    m2_list = list(minutiae2)
    n1, n2 = len(m1_list), len(m2_list)
    step1 = max(1, n1 // 12)
    step2 = max(1, n2 // 12)
    refs1 = m1_list[::step1][:12]
    refs2 = m2_list[::step2][:12]

    coords1 = np.array([[m.x, m.y] for m in m1_list], dtype=np.float64)
    coords2 = np.array([[m.x, m.y] for m in m2_list], dtype=np.float64)
    ang1 = np.array([m.orientation for m in m1_list], dtype=np.float64)
    ang2 = np.array([m.orientation for m in m2_list], dtype=np.float64)
    types1 = [m.type for m in m1_list]
    types2 = [m.type for m in m2_list]

    best = 0
    max_dist_sq = max_dist * max_dist

    for ref1 in refs1:
        for ref2 in refs2:
            if ref1.type != ref2.type:
                continue
            dtheta = ref2.orientation - ref1.orientation
            c, s = np.cos(dtheta), np.sin(dtheta)
            rel = coords1 - np.array([ref1.x, ref1.y], dtype=np.float64)
            rot = np.column_stack((c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]))
            transformed = rot + np.array([ref2.x, ref2.y], dtype=np.float64)
            t_ang = ang1 + dtheta

            used = np.zeros(n2, dtype=bool)
            matched = 0
            for i in range(n1):
                dxy = coords2 - transformed[i]
                dist_sq = dxy[:, 0] * dxy[:, 0] + dxy[:, 1] * dxy[:, 1]
                dist_sq[used] = 1e12
                j = int(np.argmin(dist_sq))
                if dist_sq[j] > max_dist_sq:
                    continue
                if types1[i] != types2[j]:
                    continue
                ad = abs((t_ang[i] - ang2[j])) % np.pi
                ad = min(ad, np.pi - ad)
                if ad <= max_angle:
                    used[j] = True
                    matched += 1
            if matched > best:
                best = matched

    score = (2.0 * best) / float(n1 + n2)
    return float(np.clip(score, 0.0, 1.0))
