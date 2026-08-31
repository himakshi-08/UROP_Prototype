"""
Minutiae Extraction Module.
Detects ridge endings and ridge bifurcations from skeletonized fingerprint images
using Crossing Number (CN) concept.
"""

import numpy as np
import cv2

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
    h, w = skeleton.shape
    minutiae = []
    
    # 8-neighbors order
    neighbors = [
        (-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1)
    ]

    for y in range(border_margin, h - border_margin):
        for x in range(border_margin, w - border_margin):
            if skeleton[y, x] != 1:
                continue
            if mask is not None and not mask[y, x]:
                continue

            # Extract 8-neighbor values
            values = [skeleton[y + dy, x + dx] for dy, dx in neighbors]
            # Crossing number
            cn = 0.5 * sum(abs(int(values[i]) - int(values[(i + 1) % 8])) for i in range(8))

            theta = 0.0
            if orientations is not None:
                theta = orientations[y, x]

            if cn == 1:
                minutiae.append(Minutia(x, y, theta, "ending"))
            elif cn == 3:
                minutiae.append(Minutia(x, y, theta, "bifurcation"))

    # Filter out spurious minutiae (too close to each other or border)
    minutiae = filter_spurious_minutiae(minutiae, distance_threshold=8.0)
    return minutiae

def filter_spurious_minutiae(minutiae, distance_threshold=8.0):
    """
    Removes false minutiae pairs that are within distance_threshold of each other.
    """
    valid = []
    n = len(minutiae)
    to_remove = set()

    for i in range(n):
        if i in to_remove:
            continue
        m1 = minutiae[i]
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            m2 = minutiae[j]
            dist = np.sqrt((m1.x - m2.x)**2 + (m1.y - m2.y)**2)
            if dist < distance_threshold:
                to_remove.add(i)
                to_remove.add(j)
                break

    for i in range(n):
        if i not in to_remove:
            valid.append(minutiae[i])

    return valid
