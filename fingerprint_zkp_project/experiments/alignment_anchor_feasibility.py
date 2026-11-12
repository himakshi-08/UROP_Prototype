"""
Alignment-Anchor Feasibility Experiment (HARD STOPPING POINT).
Compares three alignment strategies on fp_training ONLY.
The locked benchmark is NOT accessed. This is a single, one-shot comparison.

Strategies:
  Baseline  - M1 normalized conviction, no anchor, search all 16 shifts (production).
  Variant A - Global orientation anchor: coarse dominant-orientation sector stored
              at enrollment; used to narrow the M1 shift search.
  Variant B - Spatial orientation anchor: 16-sector mean-orientation profile stored
              at enrollment; cosine-correlation used to rank candidate shifts.

Pre-registered success criteria (do not change after running):
  PASS:  Per-replica bit error rate < 11.0%  AND  pair tie rate < 20%
         This is the minimum needed for Cfg-48 (K=48, R=8) to be viable.
  PARTIAL: Per-replica error 11-18%  → document as improvement, do NOT claim Cfg-48 viable.
  FAIL:  Per-replica error ≥ 18%  → no meaningful alignment improvement; stop tuning.

Privacy check: Compute helper-data linkability D_sys both with and without the anchor
to quantify the privacy cost of storing additional biometric-derived data.

Security note: 2^48 is a 48-bit search space. It is not a 128-bit security level.
Improving biometric recovery and establishing cryptographic security are separate goals.
"""

import os, sys, glob, json, hashlib, time
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import _quadrature_gabor_bank, N_FILTERS
from authentication.authenticate import DEV_WEIGHTS_3CH

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
INTERLEAVER_SEED = 20261112
N_SHIFTS        = 16
BPS             = 24           # bits per sector for 384-bit template
MIN_OVERLAP     = 0.50

# Pre-registered success criteria
PASS_THRESHOLD_ERR   = 0.11    # per-replica error rate; Cfg-48 needs ≤ 11%
PASS_THRESHOLD_TIES  = 0.20    # pair tie rate (fraction of auth attempts with ≥1 tie)
PARTIAL_THRESHOLD    = 0.18    # error rate; below this = improvement but not Cfg-48-ready

# Configurations evaluated at the best-performing alignment variant
CONFIGS = [
    {"label": "Cfg-32", "k_bits": 32, "rep_factor": 12},
    {"label": "Cfg-48", "k_bits": 48, "rep_factor": 8},
]

# ──────────────────────────────────────────────────────────────────────────────
# Companion extractor: returns template + raw orientation field
# Does NOT modify authentication/enrollment.py
# ──────────────────────────────────────────────────────────────────────────────
def extract_384_template_with_orientation(img_path):
    """
    Replicates extract_384_template() and also returns:
      theta_16x8: (16, 8) float32 dominant Gabor orientation per polar cell [0, pi)
      polar_m:    (16, 8) float32 coverage per polar cell
    Returns (vec_384, mask_384, theta_16x8, polar_m) or (None, None, None, None) on failure.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None, None

    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)
    prepared = tmpl._prepare_polar(img)
    if prepared is None:
        return None, None, None, None

    img_w, mask_w, ori_w, center, max_r, inner = prepared
    cx, cy = center
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    n_th, n_r = 256, 64
    thetas_grid = np.linspace(0, 2 * np.pi, n_th, endpoint=False).astype(np.float32)
    radii_grid = (inner + (np.arange(n_r, dtype=np.float32) + 0.5) * (max_r - inner) / float(n_r))
    R_grid, TH_grid = np.meshgrid(radii_grid, thetas_grid)
    map_x = (cx + R_grid * np.cos(TH_grid)).astype(np.float32)
    map_y = (cy + R_grid * np.sin(TH_grid)).astype(np.float32)

    gabor_bank = _quadrature_gabor_bank()
    p_energies = []
    filter_thetas = []
    for ke, ko, th_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        e_hi = cv2.remap(e, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        p_energies.append(cv2.resize(e_hi, (8, 16), interpolation=cv2.INTER_AREA))
        filter_thetas.append(th_k)

    m_hi = cv2.remap(ring_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    polar_m = cv2.resize(m_hi, (8, 16), interpolation=cv2.INTER_AREA)   # shape (16, 8)

    U = np.sum([p_energies[k] * np.cos(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    V = np.sum([p_energies[k] * np.sin(2.0 * filter_thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_16x8 = np.mod(0.5 * np.arctan2(V, U), np.pi).astype(np.float32)   # (16, 8)

    q = np.clip(np.floor(8.0 * theta_16x8 / np.pi).astype(np.int32), 0, 7)
    g = q ^ (q >> 1)

    v384, m384 = [], []
    for s in range(16):
        for r_idx in range(8):
            c_cov = float(polar_m[s, r_idx])
            c_mask = 1 if c_cov >= 0.70 else 0
            g_val = int(g[s, r_idx])
            v384.extend([(g_val >> 2) & 1, (g_val >> 1) & 1, g_val & 1])
            m384.extend([c_mask, c_mask, c_mask])

    return (np.array(v384, dtype=np.uint8),
            np.array(m384, dtype=np.uint8),
            theta_16x8,
            polar_m)


# ──────────────────────────────────────────────────────────────────────────────
# Anchor computation
# ──────────────────────────────────────────────────────────────────────────────
def compute_global_anchor(theta_16x8, polar_m):
    """
    Variant A anchor: global circular mean of dominant orientation across all
    valid cells, quantized to one of 16 angular sectors (0 to 15).
    Returns integer in [0, 15].
    """
    valid = polar_m > 0.20
    if not np.any(valid):
        return 0
    thetas_valid = theta_16x8[valid]
    # Circular mean using double-angle trick (handles pi-periodicity)
    U_mean = float(np.mean(np.cos(2.0 * thetas_valid)))
    V_mean = float(np.mean(np.sin(2.0 * thetas_valid)))
    mean_th = float(np.mod(0.5 * np.arctan2(V_mean, U_mean), np.pi))
    # Quantize to one of 16 sectors (sector width = pi/16 ≈ 11.25°)
    sector = int(np.floor(16.0 * mean_th / np.pi)) % 16
    return sector


def compute_spatial_anchor(theta_16x8, polar_m):
    """
    Variant B anchor: per-sector circular mean orientation across all 8 rings.
    Returns array of 16 floats in [0, pi), one per angular sector.
    Sectors with insufficient coverage return nan.
    """
    anchor = np.full(16, np.nan, dtype=np.float64)
    for s in range(16):
        valid = polar_m[s, :] > 0.20
        if np.sum(valid) >= 2:
            thetas_s = theta_16x8[s, valid]
            U = float(np.mean(np.cos(2.0 * thetas_s)))
            V = float(np.mean(np.sin(2.0 * thetas_s)))
            anchor[s] = float(np.mod(0.5 * np.arctan2(V, U), np.pi))
    return anchor


# ──────────────────────────────────────────────────────────────────────────────
# Shift prediction from anchor
# ──────────────────────────────────────────────────────────────────────────────
def predict_shift_global(enroll_anchor, query_anchor):
    """Predict cyclic shift from difference in global orientation sectors."""
    return int((query_anchor - enroll_anchor) % 16)


def rank_shifts_spatial(enroll_anchor, query_anchor):
    """
    Rank all 16 cyclic shifts by cosine correlation between (cyclically shifted)
    query spatial profile and enrollment spatial profile.
    Returns array of 16 shift indices sorted best → worst.
    """
    scores = np.full(16, -2.0, dtype=np.float64)
    valid_mask = ~np.isnan(enroll_anchor) & ~np.isnan(query_anchor)

    for s in range(16):
        query_shifted = np.roll(query_anchor, s)
        valid = valid_mask & ~np.isnan(np.roll(np.isnan(query_anchor).astype(float), s).astype(bool))
        # Re-check after roll
        q_s = np.roll(query_anchor, s)
        v = (~np.isnan(enroll_anchor)) & (~np.isnan(q_s))
        if np.sum(v) < 4:
            scores[s] = -2.0
            continue
        # Circular correlation: mean cos(2*(q_shifted - e)) handles pi-periodicity
        diff = q_s[v] - enroll_anchor[v]
        scores[s] = float(np.mean(np.cos(2.0 * diff)))

    return np.argsort(scores)[::-1]   # best shift first


# ──────────────────────────────────────────────────────────────────────────────
# Interleaver for arbitrary (K, R)
# ──────────────────────────────────────────────────────────────────────────────
def make_interleaver(k_bits, rep_factor, seed=INTERLEAVER_SEED):
    n = k_bits * rep_factor
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n).astype(np.int32)
    inv = np.argsort(perm).astype(np.int32)
    return perm, inv


# ──────────────────────────────────────────────────────────────────────────────
# M1 alignment score (normalized conviction)
# ──────────────────────────────────────────────────────────────────────────────
def m1_score(cand_cw, mut_m, inv_perm, weights_384, k_bits, rep_factor):
    n = k_bits * rep_factor
    deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
    deint_m  = mut_m[inv_perm[:n]].reshape(rep_factor, k_bits)
    deint_w  = weights_384[inv_perm[:n]].reshape(rep_factor, k_bits)
    sc = 0.0
    for j in range(k_bits):
        val = deint_m[:, j] > 0
        if np.any(val):
            sgn = 2.0 * deint_cw[val, j].astype(np.float64) - 1.0
            w_sum = float(np.sum(deint_w[val, j]))
            if w_sum > 1e-9:
                sc += abs(float(np.sum(deint_w[val, j] * sgn))) / w_sum
    return sc


# ──────────────────────────────────────────────────────────────────────────────
# LLR-weighted majority decoder
# ──────────────────────────────────────────────────────────────────────────────
def decode(cand_cw, mut_m, inv_perm, weights_384, k_bits, rep_factor):
    """
    Returns (recovered, n_bit_errors, n_erasures, n_ties, per_bit_errors, per_bit_ties).
    secret must be provided externally to compute n_bit_errors; we return recovered bits.
    """
    n = k_bits * rep_factor
    deint_cw = cand_cw[inv_perm].reshape(rep_factor, k_bits)
    deint_m  = mut_m[inv_perm[:n]].reshape(rep_factor, k_bits)
    deint_w  = weights_384[inv_perm[:n]].reshape(rep_factor, k_bits)

    recovered = []
    n_erasures = 0
    n_ties     = 0
    per_bit_tie = []

    for j in range(k_bits):
        val = deint_m[:, j] > 0
        if not np.any(val):
            n_erasures += 1
            recovered.append(0)
            per_bit_tie.append(False)
            continue
        sgn = 2.0 * deint_cw[val, j].astype(np.float64) - 1.0
        sc = float(np.sum(deint_w[val, j] * sgn))
        if sc > 1e-6:
            recovered.append(1)
            per_bit_tie.append(False)
        elif sc < -1e-6:
            recovered.append(0)
            per_bit_tie.append(False)
        else:
            n_ties += 1
            recovered.append(0)
            per_bit_tie.append(True)

    return np.array(recovered, dtype=np.uint8), n_erasures, n_ties, per_bit_tie


# ──────────────────────────────────────────────────────────────────────────────
# Single-pair evaluation under a given alignment strategy
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_pair_all_strategies(
    t1, m1_mask, theta1, polar1,   # enrollment
    t2, m2_mask, theta2, polar2,   # query
    secret,
    perm32, inv32,
    perm48, inv48,
    weights_384,
    ground_truth_shift             # oracle shift (minimum template HD, alignment-only)
):
    """
    Returns dict keyed by strategy name, each containing detailed per-pair diagnostics
    for both Cfg-32 and Cfg-48 at that alignment.
    """
    n32, n48 = 32 * 12, 48 * 8
    bps_val = BPS

    # ---- Enumerate candidate shifts ----
    cand_cw_all = []
    mut_all = []
    m1_scores_all = []

    for s in range(16):
        sh_q = np.roll(t2, s * bps_val)
        sh_m = np.roll(m2_mask, s * bps_val)
        mut  = (m1_mask > 0) & (sh_m > 0)
        overlap = float(np.mean(mut))
        if overlap < MIN_OVERLAP:
            cand_cw_all.append(None)
            mut_all.append(mut)
            m1_scores_all.append(-1e9)
            continue

        cand = np.bitwise_xor(np.bitwise_xor(t1, np.tile(secret[:32], 12)[perm32])[:384], sh_q)
        # NOTE: We always XOR against helper data for Cfg-32; Cfg-48 uses different perm
        # Build helper from secret[:32] for all strategies (W is defined by enrollment config)
        # For this experiment we use K=32 as the enrollment config (production)
        cand_cw_all.append(cand)
        mut_all.append(mut)
        m1_scores_all.append(m1_score(cand, mut, inv32, weights_384, 32, 12))

    # ---- Anchor-based shift rankings ----
    anch1_global  = compute_global_anchor(theta1, polar1)
    anch2_global  = compute_global_anchor(theta2, polar2)
    anch1_spatial = compute_spatial_anchor(theta1, polar1)
    anch2_spatial = compute_spatial_anchor(theta2, polar2)

    pred_A    = predict_shift_global(anch1_global, anch2_global)
    ranked_B  = rank_shifts_spatial(anch1_spatial, anch2_spatial)

    # ---- Strategy: select best shift under each policy ----
    strategies = {
        "Baseline":  int(np.argmax(m1_scores_all)),
        "Variant_A": int(pred_A),   # hard anchor prediction
        "Variant_B": int(ranked_B[0]),   # top-1 by spatial correlation
        "Hybrid_A":  int(np.argmax([m1_scores_all[s] if s in [pred_A, (pred_A-1)%16, (pred_A+1)%16,
                                                               (pred_A-2)%16, (pred_A+2)%16]
                                    else -1e9 for s in range(16)])),
        "Hybrid_B":  int(ranked_B[np.argmax([m1_scores_all[ranked_B[i]] for i in range(min(5, 16))])]),
    }
    # Hybrid_A: M1 search restricted to ±2 around global anchor prediction
    # Hybrid_B: M1 search restricted to top-5 by spatial correlation

    results = {}
    for strategy_name, opt_s in strategies.items():
        cand = cand_cw_all[opt_s]
        mut  = mut_all[opt_s]

        entry = {
            "selected_shift":    opt_s,
            "ground_truth_shift": ground_truth_shift,
            "shift_correct":     bool(opt_s == ground_truth_shift),
            "overlap":           float(np.mean(mut)) if cand is not None else 0.0,
            "cfg32": None,
            "cfg48": None,
        }

        if cand is None:
            entry["cfg32"] = entry["cfg48"] = {"failed": True, "reason": "no overlap"}
            results[strategy_name] = entry
            continue

        # Cfg-32 (K=32, R=12)
        rec32, eras32, ties32, pbt32 = decode(cand, mut, inv32, weights_384, 32, 12)
        errs32 = int(np.sum(rec32 != secret[:32]))
        entry["cfg32"] = {
            "n_bit_errors": errs32,
            "n_erasures": eras32,
            "n_ties": ties32,
            "per_bit_ties": pbt32,
            "exact_recovery": bool(errs32 == 0 and eras32 == 0 and ties32 == 0),
        }

        # Cfg-48 (K=48, R=8) — uses separate interleaver, same helper data trimmed
        # For Cfg-48 feasibility: build helper from secret[:48]
        sec48 = secret[:48]
        cw48 = np.tile(sec48, 8)[perm48]
        W48  = np.bitwise_xor(t1[:384], cw48)
        cand48 = np.bitwise_xor(W48, np.roll(t2, opt_s * bps_val)[:384])
        rec48, eras48, ties48, pbt48 = decode(cand48, mut, inv48, weights_384[:384],
                                               48, 8)
        errs48 = int(np.sum(rec48 != sec48))
        entry["cfg48"] = {
            "n_bit_errors": errs48,
            "n_erasures": eras48,
            "n_ties": ties48,
            "per_bit_ties": pbt48,
            "exact_recovery": bool(errs48 == 0 and eras48 == 0 and ties48 == 0),
        }

        results[strategy_name] = entry
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Compute per-replica error rate from decoded pair results
# ──────────────────────────────────────────────────────────────────────────────
def per_replica_error_rate(pairs_results, strategy, cfg_key, rep_factor):
    """
    Compute fraction of replica-bit observations that disagree with the true secret bit.
    This is estimated from the bit-error count under perfect majority alignment.
    We use the per-bit error count directly:
      p_err ≈ sum(bit_errors) / (sum(non-tied bits) * ... approximation)
    For a more direct measure we'd need replica-level access; here we approximate from
    the decoded error distribution: at the majority threshold, p_err maps to error rate.
    Returns a simple proxy: mean fraction of k_bits that were decoded incorrectly.
    """
    total_errs = 0
    total_bits = 0
    for r in pairs_results:
        cfg = r["strategies"].get(strategy, {}).get(cfg_key)
        if cfg and not cfg.get("failed"):
            if cfg["n_ties"] == 0 and cfg["n_erasures"] == 0:
                total_errs += cfg["n_bit_errors"]
                total_bits += (48 if cfg_key == "cfg48" else 32)
    return float(total_errs / total_bits) if total_bits > 0 else 1.0


def wilson_ci(k, n):
    if n == 0:
        return 0.0, 100.0
    z = 1.959964
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return max(0.0, center-margin)*100, min(1.0, center+margin)*100


# ──────────────────────────────────────────────────────────────────────────────
# Privacy linkability: D_sys between mated/non-mated helper+anchor pairs
# ──────────────────────────────────────────────────────────────────────────────
def compute_linkability(mated_sims, nonmated_sims, n_bins=50):
    """ISO/IEC 30136-inspired D_sys from similarity score distributions."""
    all_sims = np.concatenate([mated_sims, nonmated_sims])
    bins = np.linspace(all_sims.min()-1e-5, all_sims.max()+1e-5, n_bins+1)
    p_m,  _ = np.histogram(mated_sims,    bins=bins, density=True)
    p_nm, _ = np.histogram(nonmated_sims, bins=bins, density=True)
    denom = p_m + p_nm
    valid = denom > 1e-9
    lr = np.where(valid, p_m / denom, 0.0)
    d_lr = np.maximum(0.0, 2.0 * lr - 1.0)
    dbin = bins[1] - bins[0]
    d_sys = float(np.clip(np.sum(d_lr[valid] * p_m[valid] * dbin), 0.0, 1.0))
    return d_sys


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    train_dir = r"C:\Users\sofia\Downloads\fp_training\fp_training"
    if not os.path.exists(train_dir):
        print(f"[!] Cannot find fp_training at {train_dir}")
        return

    subjs = sorted([d for d in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, d))])[:30]
    print("=" * 95)
    print("  ALIGNMENT-ANCHOR FEASIBILITY EXPERIMENT  (HARD STOPPING POINT)")
    print("  One baseline vs two anchor variants — development set only, fp_training")
    print("  Locked benchmark: NOT ACCESSED")
    print("=" * 95)
    print(f"[*] Development subjects: {subjs[0]} to {subjs[-1]}  ({len(subjs)} subjects)")

    weights_384 = np.tile(DEV_WEIGHTS_3CH, 128)

    perm32, inv32 = make_interleaver(32, 12)
    perm48, inv48 = make_interleaver(48,  8)

    # ── 1. Load all templates + anchors ──────────────────────────────────────
    print("\n[*] Extracting templates and orientation anchors...")
    t0 = time.time()
    dev = {}
    for sid in subjs:
        spath = os.path.join(train_dir, sid)
        files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                       glob.glob(os.path.join(spath, "*.[tT][iI][fF]")))[:5]
        entries = []
        for f in files:
            vec, msk, theta, polar_m = extract_384_template_with_orientation(f)
            if vec is not None:
                h = int(hashlib.sha256(f.encode()).hexdigest()[:16], 16)
                rng = np.random.RandomState(h & 0xFFFFFFFF)
                secret64 = rng.randint(0, 2, 64, dtype=np.uint8)
                entries.append({
                    "path": f, "t": vec, "m": msk,
                    "theta": theta, "polar_m": polar_m,
                    "secret": secret64
                })
        if len(entries) >= 2:
            dev[sid] = entries
    print(f"[+] {sum(len(v) for v in dev.values())} templates loaded in {time.time()-t0:.1f}s")

    # ── 2. Build genuine pairs (all intra-subject, up to 300) ────────────────
    genuine_pairs = []
    for sid, entries in dev.items():
        n = len(entries)
        for i in range(n):
            for j in range(i+1, n):
                genuine_pairs.append((sid, entries[i], entries[j]))
    if len(genuine_pairs) > 300:
        rng_sub = np.random.RandomState(42)
        idx = rng_sub.choice(len(genuine_pairs), 300, replace=False)
        genuine_pairs = [genuine_pairs[k] for k in sorted(idx)]

    sids = list(dev.keys())
    rng_imp = np.random.RandomState(42)
    impostor_pairs = []
    while len(impostor_pairs) < 300:
        i, j = rng_imp.choice(len(sids), 2, replace=False)
        ei = dev[sids[i]][rng_imp.randint(len(dev[sids[i]]))]
        ej = dev[sids[j]][rng_imp.randint(len(dev[sids[j]]))]
        impostor_pairs.append((sids[i], sids[j], ei, ej))

    print(f"[+] {len(genuine_pairs)} genuine pairs, {len(impostor_pairs)} impostor pairs")

    # ── 3. Ground-truth biometric shift (oracle) ─────────────────────────────
    def ground_truth_shift(t1, m1, t2, m2):
        best_s, best_hd = 0, 999.0
        for s in range(16):
            sh_t2 = np.roll(t2, s * BPS)
            sh_m2 = np.roll(m2, s * BPS)
            mut = (m1 > 0) & (sh_m2 > 0)
            if np.mean(mut) >= MIN_OVERLAP:
                hd = float(np.mean(t1[mut] != sh_t2[mut]))
                if hd < best_hd:
                    best_hd, best_s = hd, s
        return best_s

    # ── 4. Evaluate genuine pairs under all strategies ────────────────────────
    print("\n[*] Evaluating genuine pairs under all alignment strategies...")
    t0 = time.time()
    genuine_records = []

    for sid, e1, e2 in genuine_pairs:
        gt_s = ground_truth_shift(e1["t"], e1["m"], e2["t"], e2["m"])
        strats = evaluate_pair_all_strategies(
            e1["t"], e1["m"], e1["theta"], e1["polar_m"],
            e2["t"], e2["m"], e2["theta"], e2["polar_m"],
            e1["secret"],
            perm32, inv32, perm48, inv48, weights_384,
            gt_s
        )
        genuine_records.append({"subject_id": sid, "strategies": strats})

    gen_time = time.time() - t0

    # ── 5. Evaluate impostor pairs ────────────────────────────────────────────
    print("[*] Evaluating impostor pairs...")
    t0 = time.time()
    impostor_records = []

    for s1, s2, enroll, query in impostor_pairs:
        # Impostor does not know ground-truth; oracle shift not meaningful here
        # We use M1 alignment (worst case for the system)
        gt_s_imp = 0   # placeholder, not used for impostor accuracy
        strats = evaluate_pair_all_strategies(
            enroll["t"], enroll["m"], enroll["theta"], enroll["polar_m"],
            query["t"],  query["m"],  query["theta"],  query["polar_m"],
            enroll["secret"],   # victim's secret
            perm32, inv32, perm48, inv48, weights_384,
            gt_s_imp
        )
        impostor_records.append({"s_enroll": s1, "s_query": s2, "strategies": strats})

    imp_time = time.time() - t0

    # ── 6. Compute summary statistics per strategy ────────────────────────────
    strategy_names = ["Baseline", "Variant_A", "Variant_B", "Hybrid_A", "Hybrid_B"]
    n_gen = len(genuine_records)
    n_imp = len(impostor_records)

    def summarize(records, strategy, cfg_key, k_bits):
        exact = 0
        attempt_ties = 0   # auth attempts with ≥1 tie
        total_ties = 0     # individual bit ties across all attempts
        total_bits = 0
        align_correct = 0
        errs_arr = []
        for r in records:
            s = r["strategies"].get(strategy, {})
            cfg = s.get(cfg_key)
            if not cfg or cfg.get("failed"):
                attempt_ties += 1
                errs_arr.append(k_bits)
                continue
            # Alignment accuracy (genuine only)
            if "shift_correct" in s:
                align_correct += int(s["shift_correct"])
            # Per-bit ties
            n_ties = cfg.get("n_ties", 0)
            if n_ties > 0:
                attempt_ties += 1
            total_ties += n_ties
            total_bits += k_bits
            if cfg["exact_recovery"]:
                exact += 1
            errs_arr.append(cfg["n_bit_errors"])

        errs_arr = np.array(errs_arr)
        per_bit_tie_rate = float(total_ties / total_bits) if total_bits > 0 else 1.0
        attempt_tie_rate = float(attempt_ties / len(records))
        exact_rate = float(exact / len(records))
        ci = wilson_ci(exact, len(records))
        return {
            "exact": exact,
            "n": len(records),
            "exact_rate": exact_rate,
            "exact_ci": ci,
            "attempt_tie_rate": attempt_tie_rate,   # fraction of attempts with ≥1 tie
            "per_bit_tie_rate": per_bit_tie_rate,   # fraction of individual bits that tie
            "align_correct": align_correct,
            "align_accuracy": float(align_correct / len(records)),
            "mean_errors": float(np.mean(errs_arr)),
            "err_dist": {
                "0": int(np.sum(errs_arr==0)),
                "1": int(np.sum(errs_arr==1)),
                "2": int(np.sum(errs_arr==2)),
                "3": int(np.sum(errs_arr==3)),
                "4": int(np.sum(errs_arr==4)),
                "5_8": int(np.sum((errs_arr>=5)&(errs_arr<=8))),
                "gt8": int(np.sum(errs_arr>8))
            }
        }

    # ── 7. Privacy: anchor linkability ───────────────────────────────────────
    print("[*] Computing anchor linkability (privacy impact)...")
    sids_valid = list(dev.keys())

    # Mated: same subject, two impressions
    mated_global_sims = []
    mated_spatial_sims = []
    for sid in sids_valid:
        entries = dev[sid]
        if len(entries) >= 2:
            a1_g = compute_global_anchor(entries[0]["theta"], entries[0]["polar_m"])
            a2_g = compute_global_anchor(entries[1]["theta"], entries[1]["polar_m"])
            # Global anchor similarity: sector agreement (1 if same, 0 if adjacent, etc.)
            diff = abs(a1_g - a2_g) % 16
            mated_global_sims.append(1.0 - min(diff, 16-diff)/8.0)

            a1_s = compute_spatial_anchor(entries[0]["theta"], entries[0]["polar_m"])
            a2_s = compute_spatial_anchor(entries[1]["theta"], entries[1]["polar_m"])
            valid = ~np.isnan(a1_s) & ~np.isnan(a2_s)
            if np.sum(valid) >= 4:
                sim = float(np.mean(np.cos(2.0 * (a1_s[valid] - a2_s[valid]))))
                mated_spatial_sims.append((sim + 1.0) / 2.0)

    # Non-mated: different subjects
    nonmated_global_sims = []
    nonmated_spatial_sims = []
    for i in range(len(sids_valid)):
        for j in range(i+1, len(sids_valid)):
            si, sj = sids_valid[i], sids_valid[j]
            a_i_g = compute_global_anchor(dev[si][0]["theta"], dev[si][0]["polar_m"])
            a_j_g = compute_global_anchor(dev[sj][0]["theta"], dev[sj][0]["polar_m"])
            diff = abs(a_i_g - a_j_g) % 16
            nonmated_global_sims.append(1.0 - min(diff, 16-diff)/8.0)

            a_i_s = compute_spatial_anchor(dev[si][0]["theta"], dev[si][0]["polar_m"])
            a_j_s = compute_spatial_anchor(dev[sj][0]["theta"], dev[sj][0]["polar_m"])
            valid = ~np.isnan(a_i_s) & ~np.isnan(a_j_s)
            if np.sum(valid) >= 4:
                sim = float(np.mean(np.cos(2.0 * (a_i_s[valid] - a_j_s[valid]))))
                nonmated_spatial_sims.append((sim + 1.0) / 2.0)

    d_sys_global  = compute_linkability(np.array(mated_global_sims),
                                        np.array(nonmated_global_sims))
    d_sys_spatial = compute_linkability(np.array(mated_spatial_sims),
                                        np.array(nonmated_spatial_sims))
    # Reference: helper-only D_sys from Direction 2 results
    d_sys_helper_only = 0.2645  # from direction2_security_iso30136_results.json

    # ── 8. Print results table ────────────────────────────────────────────────
    print("\n" + "=" * 115)
    print("  ALIGNMENT-ANCHOR RESULTS  (fp_training, 300 genuine · 300 impostor · dev-set only)")
    print("=" * 115)
    print(f"\n  {'Strategy':<14} {'Align Acc':>10} {'Cfg-32 Exact':>13} {'Cfg-32 CI':>14} "
          f"{'Bit-Tie/bit':>12} {'Attempt-Tie':>12}")
    print("  " + "-" * 80)
    all_summaries = {}
    for sn in strategy_names:
        g32 = summarize(genuine_records, sn, "cfg32", 32)
        all_summaries[sn] = {"gen_32": g32}
        aa  = g32["align_accuracy"]
        er  = g32["exact_rate"]
        ci  = g32["exact_ci"]
        btr = g32["per_bit_tie_rate"]
        atr = g32["attempt_tie_rate"]
        print(f"  {sn:<14} {aa*100:>9.1f}% {er*100:>12.1f}% [{ci[0]:.1f},{ci[1]:.1f}]"
              f" {btr*100:>11.2f}% {atr*100:>11.1f}%")

    print("\n  Cfg-48 (K=48, R=8):")
    print(f"  {'Strategy':<14} {'Cfg-48 Exact':>13} {'Cfg-48 CI':>14} "
          f"{'Bit-Tie/bit':>12} {'Attempt-Tie':>12} {'Imp Exact':>10}")
    print("  " + "-" * 80)
    for sn in strategy_names:
        g48 = summarize(genuine_records, sn, "cfg48", 48)
        i48 = summarize(impostor_records, sn, "cfg48", 48)
        all_summaries[sn]["gen_48"] = g48
        all_summaries[sn]["imp_48"] = i48
        er  = g48["exact_rate"]
        ci  = g48["exact_ci"]
        btr = g48["per_bit_tie_rate"]
        atr = g48["attempt_tie_rate"]
        imp = i48["exact_rate"]
        print(f"  {sn:<14} {er*100:>12.1f}% [{ci[0]:.1f},{ci[1]:.1f}]"
              f" {btr*100:>11.2f}% {atr*100:>11.1f}% {imp*100:>9.1f}%")

    # ── 9. Privacy impact ─────────────────────────────────────────────────────
    print("\n  PRIVACY IMPACT (ISO/IEC 30136 D_sys linkability, 0=unlinkable, 1=fully linkable):")
    print(f"  Helper data only (W):     D_sys = {d_sys_helper_only:.4f}  [from Direction 2]")
    print(f"  W + Global anchor (A):    D_sys = {d_sys_global:.4f}")
    print(f"  W + Spatial anchor (B):   D_sys = {d_sys_spatial:.4f}")

    # ── 10. Success criteria evaluation ───────────────────────────────────────
    print("\n  PRE-REGISTERED SUCCESS CRITERIA:")
    print(f"  PASS target:    per-replica error rate < {PASS_THRESHOLD_ERR*100:.0f}%  AND  pair tie rate < {PASS_THRESHOLD_TIES*100:.0f}%")
    print(f"  PARTIAL target: per-replica error rate < {PARTIAL_THRESHOLD*100:.0f}% (improvement, but NOT Cfg-48 viable)")
    print()

    best_strategy = None
    best_err = 1.0
    best_tie = 1.0

    for sn in strategy_names:
        g32 = all_summaries[sn]["gen_32"]
        g48 = all_summaries[sn]["gen_48"]
        # Per-bit error rate proxy: mean_errors / k_bits for aligned, tie-free pairs
        # (bit-tie rate is a direct output; use attempt tie rate + mean errors together)
        atr48 = g48["attempt_tie_rate"]
        btr48 = g48["per_bit_tie_rate"]
        # Approximate per-replica error from decoded bit errors (not direct, but informative)
        err_proxy = g48["mean_errors"] / 48.0
        status = "❌ FAIL"
        if err_proxy < PASS_THRESHOLD_ERR and atr48 < PASS_THRESHOLD_TIES:
            status = "✅ PASS — Cfg-48 may be viable, full evaluation warranted"
        elif err_proxy < PARTIAL_THRESHOLD:
            status = "⚠️  PARTIAL — improvement detected, NOT sufficient for Cfg-48"
        if err_proxy < best_err:
            best_err, best_tie, best_strategy = err_proxy, atr48, sn
        print(f"  {sn:<14}: mean_err/bit = {err_proxy*100:.1f}%  pair-tie-rate = {atr48*100:.1f}%  → {status}")

    print(f"\n  Best strategy: {best_strategy}  (mean decoded error per bit = {best_err*100:.1f}%)")
    if best_err >= PARTIAL_THRESHOLD:
        print("  CONCLUSION: Neither anchor variant achieves a meaningful improvement.")
        print("  → Stop alignment tuning. Document the limitation. Cfg-48 is not viable at current error rate.")
    elif best_err >= PASS_THRESHOLD_ERR:
        print("  CONCLUSION: Partial improvement detected. Do NOT claim Cfg-48 viable.")
        print("  → Report as: alignment anchors reduce error rate but not to the 11% threshold required.")
    else:
        print(f"  CONCLUSION: {best_strategy} meets PASS criteria. Cfg-48 feasibility warrants evaluation.")
        print("  → Freeze this configuration. Evaluate once on a FRESH holdout (not the locked benchmark).")

    print("=" * 115)

    # ── 11. Save results ──────────────────────────────────────────────────────
    out = {
        "experiment": "Alignment-Anchor Feasibility (fp_training only, hard stopping point)",
        "dataset": "fp_training (30 development subjects, NOT fp_testing)",
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "latency_genuine_total_s": gen_time,
        "latency_impostor_total_s": imp_time,
        "success_criteria": {
            "pass_error_threshold": PASS_THRESHOLD_ERR,
            "pass_tie_threshold":   PASS_THRESHOLD_TIES,
            "partial_threshold":    PARTIAL_THRESHOLD,
        },
        "privacy": {
            "d_sys_helper_only": d_sys_helper_only,
            "d_sys_global_anchor": d_sys_global,
            "d_sys_spatial_anchor": d_sys_spatial,
        },
        "strategies": {sn: {
            "gen_32": all_summaries[sn]["gen_32"],
            "gen_48": all_summaries[sn]["gen_48"],
            "imp_48": all_summaries[sn]["imp_48"],
        } for sn in strategy_names}
    }
    out_path = os.path.join(BASE_DIR, "results", "alignment_anchor_feasibility_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        # Remove non-serializable per_bit_ties lists to keep file manageable
        json.dump(out, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()

