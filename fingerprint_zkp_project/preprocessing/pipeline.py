"""
Unified Fingerprint Preprocessing Pipeline.
Combines Normalization, Segmentation, Gabor Enhancement, Binarization, and Skeletonization.
"""

from .normalization import normalize
from .segmentation import create_segmentation_mask
from .enhancement import estimate_orientations, apply_gabor_enhancement
from .thinning import binarize, skeletonize_ridges

def preprocess_fingerprint(image, m0=100.0, v0=100.0, block_size=16, gabor_ksize=15):
    """
    Executes complete preprocessing pipeline on raw grayscale fingerprint.
    
    Returns:
        dict containing intermediate and final preprocessed outputs:
            - normalized
            - segmented
            - mask
            - orientations
            - enhanced
            - binary
            - skeleton
    """
    norm = normalize(image, m0=m0, v0=v0)
    seg, mask = create_segmentation_mask(norm, block_size=block_size)
    orientations = estimate_orientations(seg, block_size=block_size)
    enhanced = apply_gabor_enhancement(seg, orientations, ksize=gabor_ksize)
    
    # Mask background out of enhanced
    enhanced[~mask] = 255
    
    binary = binarize(enhanced)
    binary[~mask] = 0
    
    skeleton = skeletonize_ridges(binary)
    skeleton[~mask] = 0
    
    return {
        "normalized": norm,
        "segmented": seg,
        "mask": mask,
        "orientations": orientations,
        "enhanced": enhanced,
        "binary": binary,
        "skeleton": skeleton
    }
