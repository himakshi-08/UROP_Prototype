"""
Fingerprint Enhancement Module.
Computes ridge orientation fields and applies Gabor filtering for noise reduction and ridge enhancement.
"""

import numpy as np
import cv2

def estimate_orientations(image, block_size=16):
    """
    Estimates local ridge orientation using gradient-based Sobel operators.
    
    Returns:
        np.ndarray: Orientation map in radians (-pi/2 to pi/2).
    """
    image_f = image.astype(np.float64)
    gx = cv2.Sobel(image_f, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_f, cv2.CV_64F, 0, 1, ksize=3)

    gxx = gx**2
    gyy = gy**2
    gxy = gx * gy

    h, w = image.shape
    orientations = np.zeros_like(image_f)

    # Smooth gradients over blocks
    kernel = np.ones((block_size, block_size))
    vx = cv2.filter2D(2 * gxy, -1, kernel)
    vy = cv2.filter2D(gxx - gyy, -1, kernel)

    orientations = 0.5 * np.arctan2(vx, vy) + (np.pi / 2)
    return orientations

def apply_gabor_enhancement(image, orientations, wavelength=8.0, ksize=15):
    """
    Enhances fingerprint ridges using oriented Gabor filtering.
    """
    h, w = image.shape
    enhanced = np.zeros_like(image, dtype=np.float64)
    
    # Pre-generate bank of Gabor filters for discrete angles (0 to 180 degrees in steps of 10)
    num_angles = 18
    gabor_bank = []
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)

    for theta in angles:
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=4.0,
            theta=theta + np.pi/2, # perpendicular to ridge orientation
            lambd=wavelength,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_64F
        )
        gabor_bank.append(kernel)

    # Apply filter based on nearest orientation index
    angle_indices = np.round(orientations / (np.pi / num_angles)).astype(int) % num_angles

    for idx in range(num_angles):
        mask = (angle_indices == idx)
        if np.any(mask):
            filtered = cv2.filter2D(image.astype(np.float64), -1, gabor_bank[idx])
            enhanced[mask] = filtered[mask]

    # Normalize enhanced image back to 0-255
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return enhanced
