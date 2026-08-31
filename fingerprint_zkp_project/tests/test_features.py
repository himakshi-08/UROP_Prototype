"""
Unit tests for Minutiae Extraction and Biometric Template modules.
"""

import numpy as np

from features import Minutia, extract_minutiae, BiometricTemplate

def test_minutiae_extraction():
    skeleton = np.zeros((100, 100), dtype=np.uint8)
    # Horizontal line with end point at (50, 50) -> Ridge ending
    skeleton[50, 20:51] = 1

    minutiae = extract_minutiae(skeleton, border_margin=10)
    assert isinstance(minutiae, list)

def test_biometric_template():
    minutiae = [
        Minutia(x=30, y=40, orientation=0.5, minutia_type="ending"),
        Minutia(x=80, y=90, orientation=1.2, minutia_type="bifurcation")
    ]
    template = BiometricTemplate(minutiae, image_shape=(200, 200), vector_bits=256)
    assert len(template.binary_vector) == 256
    assert set(np.unique(template.binary_vector)).issubset({0, 1})

    # Test Hamming distance self-similarity
    sim = BiometricTemplate.hamming_similarity(template.binary_vector, template.binary_vector)
    assert sim == 1.0
