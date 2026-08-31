"""
Unit tests for Fingerprint Preprocessing modules.
"""

import numpy as np

from preprocessing import (
    normalize,
    create_segmentation_mask,
    estimate_orientations,
    apply_gabor_enhancement,
    binarize,
    skeletonize_ridges,
    preprocess_fingerprint
)

def test_normalization():
    img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    norm = normalize(img, m0=100.0, v0=100.0)
    assert norm.shape == (100, 100)
    assert norm.dtype == np.uint8

def test_segmentation():
    img = np.ones((100, 100), dtype=np.uint8) * 128
    img[20:80, 20:80] = np.random.randint(0, 255, (60, 60), dtype=np.uint8)
    seg, mask = create_segmentation_mask(img, block_size=16)
    assert seg.shape == (100, 100)
    assert mask.dtype == bool

def test_orientation_and_enhancement():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    orientations = estimate_orientations(img)
    enhanced = apply_gabor_enhancement(img, orientations)
    assert orientations.shape == (100, 100)
    assert enhanced.shape == (100, 100)

def test_binarize_and_skeletonize():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    binary = binarize(img)
    skeleton = skeletonize_ridges(binary)
    assert binary.shape == (100, 100)
    assert set(np.unique(binary)).issubset({0, 1})
    assert set(np.unique(skeleton)).issubset({0, 1})

def test_full_preprocessing_pipeline():
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    res = preprocess_fingerprint(img)
    assert "normalized" in res
    assert "skeleton" in res
    assert res["skeleton"].shape == (128, 128)
