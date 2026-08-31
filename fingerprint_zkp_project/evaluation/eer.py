"""
Equal Error Rate (EER) Evaluation Module.
Finds operating threshold where False Acceptance Rate equals False Rejection Rate.
"""

import numpy as np
from .far_frr import compute_far_frr

def compute_eer(genuine_scores, impostor_scores):
    """
    Computes EER value and EER operating threshold.
    
    Returns:
        dict containing:
            - eer (float): Equal Error Rate value.
            - threshold (float): Optimal decision threshold.
            - far_at_eer (float)
            - frr_at_eer (float)
    """
    metrics = compute_far_frr(genuine_scores, impostor_scores, thresholds=np.linspace(0.0, 1.0, 1001))
    thresholds = metrics["thresholds"]
    far = metrics["far"]
    frr = metrics["frr"]

    # Point of minimal difference between FAR and FRR
    diff = np.abs(far - frr)
    idx = np.argmin(diff)

    eer = (far[idx] + frr[idx]) / 2.0
    optimal_threshold = thresholds[idx]

    return {
        "eer": float(eer),
        "threshold": float(optimal_threshold),
        "far_at_eer": float(far[idx]),
        "frr_at_eer": float(frr[idx]),
        "far_curve": far,
        "frr_curve": frr,
        "thresholds": thresholds
    }
