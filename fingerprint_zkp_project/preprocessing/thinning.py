"""
Fingerprint Binarization & Skeletonization (Thinning) Module.
Converts grayscale enhanced image into binary, and thins ridges to 1-pixel wide skeleton.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize

def binarize(image, block_size=15, c=2):
    """
    Adaptive thresholding binarization of enhanced fingerprint image.
    
    Returns:
        np.ndarray: Binary image (0 for background, 1 for ridges).
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, block_size, c
    )
    return (binary > 0).astype(np.uint8)

def skeletonize_ridges(binary_image):
    """
    Applies morphological thinning to reduce binary ridges to 1-pixel wide skeletons.
    
    Args:
        binary_image (np.ndarray): Binary image (1 for ridges, 0 for background).
        
    Returns:
        np.ndarray: Skeletonized binary image (1 for 1-pixel wide skeleton, 0 otherwise).
    """
    skeleton = skeletonize(binary_image > 0)
    return skeleton.astype(np.uint8)
