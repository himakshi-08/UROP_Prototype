"""
ROC Curve Evaluation & Visualizer Module.
Computes ROC curves (TAR vs FAR) and calculates Area Under Curve (AUC).
"""

import numpy as np

def compute_roc(far, tar):
    """
    Computes Area Under Curve (AUC) for ROC plot.
    """
    # Ensure sorted by FAR ascending
    sorted_indices = np.argsort(far)
    far_sorted = far[sorted_indices]
    tar_sorted = tar[sorted_indices]
    
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(tar_sorted, far_sorted)
    else:
        from scipy.integrate import trapezoid
        auc = trapezoid(tar_sorted, far_sorted)

    return {
        "far": far_sorted,
        "tar": tar_sorted,
        "auc": float(auc)
    }
