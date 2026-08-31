"""
Fingerprint Normalization Module.
Scales image gray level values to specified desired mean M0 and variance V0
to standardize contrast across different sensor captures.
"""

import numpy as np

def normalize(image, m0=100.0, v0=100.0):
    """
    Normalizes the gray levels of a fingerprint image.
    
    Args:
        image (np.ndarray): Input 2D grayscale image (0-255).
        m0 (float): Target mean.
        v0 (float): Target variance.
        
    Returns:
        np.ndarray: Normalized grayscale image (uint8).
    """
    image = image.astype(np.float64)
    m = np.mean(image)
    v = np.var(image)
    
    if v == 0:
        return image.astype(np.uint8)

    # Pixel-wise normalization based on variance thresholding
    normalized = np.zeros_like(image)
    mask = image > m
    
    normalized[mask] = m0 + np.sqrt((v0 * (image[mask] - m)**2) / v)
    normalized[~mask] = m0 - np.sqrt((v0 * (image[~mask] - m)**2) / v)
    
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized
