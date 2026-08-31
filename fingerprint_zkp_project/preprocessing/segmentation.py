"""
Fingerprint Segmentation Module.
Extracts Region of Interest (ROI) by separating foreground ridge regions from background.
"""

import numpy as np
import cv2

def create_segmentation_mask(image, block_size=16, threshold=0.1):
    """
    Computes block-wise normalized variance to segment foreground fingerprint region from background.
    
    Args:
        image (np.ndarray): Grayscale fingerprint image.
        block_size (int): Size of local blocks (default: 16).
        threshold (float): Relative variance threshold for foreground classification.
        
    Returns:
        tuple: (segmented_image, binary_mask)
    """
    h, w = image.shape
    image_std = np.zeros_like(image, dtype=np.float64)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:min(i+block_size, h), j:min(j+block_size, w)]
            image_std[i:min(i+block_size, h), j:min(j+block_size, w)] = np.std(block)

    # Normalize std matrix
    max_std = np.max(image_std)
    if max_std > 0:
        image_std = image_std / max_std

    mask = image_std > threshold
    
    # Morphological cleaning (closing then opening)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size))
    mask_clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel).astype(bool)

    segmented = image.copy()
    segmented[~mask_clean] = 255 # Mask background to white

    return segmented, mask_clean
