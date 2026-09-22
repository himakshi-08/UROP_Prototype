"""
Minutiae Extraction Module.
Detects ridge endings and ridge bifurcations from skeletonized fingerprint images
using Crossing Number (CN) concept.
"""

import numpy as np


class Minutia:
    def __init__(self, x, y, orientation, minutia_type):
        self.x = int(x)
        self.y = int(y)
        self.orientation = float(orientation)  # in radians
        self.type = str(minutia_type)  # "ending" or "bifurcation"

    def to_tuple(self):
        return (self.x, self.y, self.orientation, self.type)

    def __repr__(self):
        return f"Minutia(x={self.x}, y={self.y}, theta={self.orientation:.2f}, type='{self.type}')"


def extract_minutiae(skeleton, orientations=None, mask=None, border_margin=15):
    """
    Extracts minutiae using Crossing Number (CN) on 3x3 window of skeletonized image.
    CN = 0.5 * sum(|P_i - P_{i+1}|) over 8-neighbors.
    - CN == 1: Ridge Ending
    - CN == 3: Ridge Bifurcation
    """
    sk = (np.asarray(skeleton) == 1).astype(np.uint8)
    h, w = sk.shape
    if h < 2 * border_margin + 3 or w < 2 * border_margin + 3:
        return []

    offsets = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1),
    ]
    neighbors = []
    for dy, dx in offsets:
        n = np.zeros_like(sk)
        y0_src, y1_src = max(0, -dy), min(h, h - dy)
        x0_src, x1_src = max(0, -dx), min(w, w - dx)
        y0_dst, x0_dst = y0_src + dy, x0_src + dx
        n[y0_dst:y0_dst + (y1_src - y0_src), x0_dst:x0_dst + (x1_src - x0_src)] = sk[y0_src:y1_src, x0_src:x1_src]
        neighbors.append(n.astype(np.int16))

    cn = np.zeros((h, w), dtype=np.float32)
    for i in range(8):
        cn += np.abs(neighbors[i] - neighbors[(i + 1) % 8])
    cn *= 0.5

    ridge = sk.astype(bool)
    if mask is not None:
        ridge &= np.asarray(mask).astype(bool)
    ridge[:border_margin, :] = False
    ridge[-border_margin:, :] = False
    ridge[:, :border_margin] = False
    ridge[:, -border_margin:] = False

    ending_ys, ending_xs = np.where(ridge & np.isclose(cn, 1.0))
    bif_ys, bif_xs = np.where(ridge & np.isclose(cn, 3.0))

    minutiae = []
    for y, x in zip(ending_ys, ending_xs):
        theta = float(orientations[y, x]) if orientations is not None else 0.0
        minutiae.append(Minutia(x, y, theta, "ending"))
    for y, x in zip(bif_ys, bif_xs):
        theta = float(orientations[y, x]) if orientations is not None else 0.0
        minutiae.append(Minutia(x, y, theta, "bifurcation"))

    return filter_spurious_minutiae(minutiae, distance_threshold=8.0)


def filter_spurious_minutiae(minutiae, distance_threshold=8.0):
    """
    Removes false minutiae pairs that are within distance_threshold of each other.
    """
    n = len(minutiae)
    if n == 0:
        return []

    coords = np.array([[m.x, m.y] for m in minutiae], dtype=np.float64)
    to_remove = set()
    thresh_sq = distance_threshold * distance_threshold

    for i in range(n):
        if i in to_remove:
            continue
        dxy = coords[i + 1:] - coords[i]
        dist_sq = np.sum(dxy * dxy, axis=1)
        close = np.where(dist_sq < thresh_sq)[0]
        if close.size:
            to_remove.add(i)
            for j in close:
                to_remove.add(i + 1 + int(j))

    return [m for i, m in enumerate(minutiae) if i not in to_remove]
