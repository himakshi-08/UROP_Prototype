"""
Development Set Tuning for Step 4 (Test-Set Isolation Protocol).
Uses 50 genuine pairs and 50 impostor pairs strictly drawn from fp_training
(completely disjoint from fp_testing benchmark).
Evaluates delta in {2 deg, 4 deg, 6 deg, 8 deg, 10 deg} on B1 (512-bit) and B2 (256-bit).
Selects delta* based on the trade-off between separation, d', AUC, and valid overlap (>50%).
"""

import os
import sys
import glob
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.dataset_loader import default_split_dirs
from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from features.template import _quadrature_gabor_bank, N_FILTERS, N_SECTORS, N_RINGS

def d_pi(a, b):
    diff = np.abs(a - b) % np.pi
    return np.minimum(diff, np.pi - diff)

def d_boundary(theta):
    boundaries = np.arange(8) * (np.pi / 8.0)
    diffs = [d_pi(theta, b) for b in boundaries]
    return np.min(diffs, axis=0)

def extract_step4_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    prep = preprocess_fingerprint(img)
    minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
    tmpl = build_template(img, prep, minutiae, vector_bits=256)

    prepared = tmpl._prepare_polar(img)
    if prepared is None:
        return None
    img_w, mask_w, ori_w, center, max_r, inner = prepared
    h, w = img_w.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
    img_roi = img_w * ring_mask

    n_sec, n_ring = 16, 8
    dsize = (n_ring, n_sec)
    flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_AREA

    # 8-orientation quadrature Gabor responses
    gabor_bank = _quadrature_gabor_bank()
    p_energies = []
    thetas = []
    for ke, ko, theta_k in gabor_bank:
        re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
        ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
        e = np.sqrt(re * re + ro * ro) * ring_mask
        p_energies.append(cv2.warpPolar(e, dsize, center, max_r, flags))
        thetas.append(theta_k)

    polar_m = cv2.warpPolar(ring_mask, dsize, center, max_r, flags)
    valid_cells = (polar_m > 0.2)

    # 1. Energy & Orientation for 128 cells (16x8)
    E_total_128 = np.sum(p_energies, axis=0) # (16, 8)
    med_energy_128 = float(np.median(E_total_128[valid_cells])) if np.any(valid_cells) else float(np.median(E_total_128))
    bits_e_128 = (E_total_128 > med_energy_128).astype(np.uint8)

    U_128 = np.sum([p_energies[k] * np.cos(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    V_128 = np.sum([p_energies[k] * np.sin(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_128 = np.mod(0.5 * np.arctan2(V_128, U_128), np.pi).astype(np.float32)

    # 3-bit Gray code per cell:
    q_128 = np.clip(np.floor(8.0 * theta_128 / np.pi).astype(np.int32), 0, 7)
    g_128 = q_128 ^ (q_128 >> 1) # Gray code: 0..7

    # Distance to nearest boundary for continuous orientation
    d_bound_128 = d_boundary(theta_128)

    # Build 512-bit representation (128 cells x 4 bits = 512 bits)
    # Cell c: [Energy, Gray_b2, Gray_b1, Gray_b0]
    vec_512 = []
    cov_512 = []
    dbound_512 = []
    for s in range(n_sec):
        for r in range(n_ring):
            c_cov = float(polar_m[s, r])
            c_db = float(d_bound_128[s, r])
            g_val = int(g_128[s, r])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1
            e_bit = int(bits_e_128[s, r])

            vec_512.extend([e_bit, b2, b1, b0])
            cov_512.extend([c_cov, c_cov, c_cov, c_cov])
            dbound_512.extend([999.0, c_db, c_db, c_db]) # Energy bit is independent of orientation boundary

    # 2. Pooled 64 cells (16 sectors x 4 pooled rings = 64 cells x 4 bits = 256 bits)
    # Pool rings (0,1), (2,3), (4,5), (6,7)
    vec_256_b2 = []
    cov_256_b2 = []
    dbound_256_b2 = []

    p_energies_pooled = []
    for k in range(N_FILTERS):
        pe = p_energies[k]
        pe_p = np.zeros((16, 4), dtype=np.float32)
        for p in range(4):
            pe_p[:, p] = 0.5 * (pe[:, 2*p] + pe[:, 2*p+1])
        p_energies_pooled.append(pe_p)

    polar_m_pooled = np.zeros((16, 4), dtype=np.float32)
    for p in range(4):
        polar_m_pooled[:, p] = 0.5 * (polar_m[:, 2*p] + polar_m[:, 2*p+1])

    valid_cells_pooled = (polar_m_pooled > 0.2)
    E_total_64 = np.sum(p_energies_pooled, axis=0)
    med_energy_64 = float(np.median(E_total_64[valid_cells_pooled])) if np.any(valid_cells_pooled) else float(np.median(E_total_64))
    bits_e_64 = (E_total_64 > med_energy_64).astype(np.uint8)

    U_64 = np.sum([p_energies_pooled[k] * np.cos(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    V_64 = np.sum([p_energies_pooled[k] * np.sin(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
    theta_64 = np.mod(0.5 * np.arctan2(V_64, U_64), np.pi).astype(np.float32)
    q_64 = np.clip(np.floor(8.0 * theta_64 / np.pi).astype(np.int32), 0, 7)
    g_64 = q_64 ^ (q_64 >> 1)
    d_bound_64 = d_boundary(theta_64)

    for s in range(16):
        for p in range(4):
            c_cov = float(polar_m_pooled[s, p])
            c_db = float(d_bound_64[s, p])
            g_val = int(g_64[s, p])
            b2 = (g_val >> 2) & 1
            b1 = (g_val >> 1) & 1
            b0 = (g_val >> 0) & 1
            e_bit = int(bits_e_64[s, p])

            vec_256_b2.extend([e_bit, b2, b1, b0])
            cov_256_b2.extend([c_cov, c_cov, c_cov, c_cov])
            dbound_256_b2.extend([999.0, c_db, c_db, c_db])

    return {
        "b1_512": {
            "vec": np.array(vec_512, dtype=np.uint8),
            "cov": np.array(cov_512, dtype=np.float32),
            "dbound": np.array(dbound_512, dtype=np.float32),
            "bits_per_sector": 32
        },
        "b2_256": {
            "vec": np.array(vec_256_b2, dtype=np.uint8),
            "cov": np.array(cov_256_b2, dtype=np.float32),
            "dbound": np.array(dbound_256_b2, dtype=np.float32),
            "bits_per_sector": 16
        }
    }

def masked_hd(v1, v2, m1, m2, min_overlap=0.50):
    m = (m1 > 0) & (m2 > 0)
    n_valid = int(np.sum(m))
    total_bits = len(v1)
    overlap = n_valid / float(total_bits) if total_bits > 0 else 0.0
    if overlap < min_overlap or n_valid == 0:
        return None
    errs = int(np.sum(v1[m] != v2[m]))
    norm_hd = errs / float(n_valid)
    return {
        "norm_hd": norm_hd,
        "equiv_bits": norm_hd * float(total_bits),
        "errs": errs,
        "n_valid": n_valid,
        "overlap": overlap
    }

def aligned_masked_hd(v1, v2, m1, m2, bits_per_sector, n_sectors=16, min_overlap=0.50):
    best = None
    for s in range(n_sectors):
        sh_v2 = np.roll(v2, s * bits_per_sector)
        sh_m2 = np.roll(m2, s * bits_per_sector)
        res = masked_hd(v1, sh_v2, m1, sh_m2, min_overlap=min_overlap)
        if res is not None:
            if best is None or res["norm_hd"] < best["norm_hd"]:
                best = dict(res)
                best["shift"] = s
    return best

def compute_auc(gen_scores, imp_scores):
    all_scores = np.concatenate([gen_scores, imp_scores])
    thresholds = np.linspace(np.min(all_scores) - 1e-4, np.max(all_scores) + 1e-4, 1001)
    tpr = [np.mean(np.array(gen_scores) >= tau) for tau in thresholds]
    fpr = [np.mean(np.array(imp_scores) >= tau) for tau in thresholds]
    s_idx = np.argsort(fpr)
    s_fpr = np.array(fpr)[s_idx]
    s_tpr = np.array(tpr)[s_idx]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz_func(s_tpr, s_fpr))

def main():
    splits = default_split_dirs()
    train_dir = splits["train"]
    print(f"[*] Extracting development set from fp_training: {train_dir}")

    subjs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])[:30]

    dev_cache = {}
    for sid in subjs:
        spath = os.path.join(train_dir, sid)
        files = sorted(glob.glob(os.path.join(spath, "*.[bB][mM][pP]")) +
                       glob.glob(os.path.join(spath, "*.[tT][iI][fF]")) +
                       glob.glob(os.path.join(spath, "*.[pP][nN][gG]")))
        if len(files) >= 2:
            tmpls = []
            for f in files[:3]:
                feat = extract_step4_features(f)
                if feat is not None:
                    tmpls.append((os.path.basename(f), feat))
            if len(tmpls) >= 2:
                dev_cache[sid] = tmpls

    print(f"[+] Loaded development templates across {len(dev_cache)} training subjects.")

    # Build 50 genuine and 50 impostor development pairs
    dev_genuine = []
    for sid, tmpls in dev_cache.items():
        n = len(tmpls)
        for i in range(n):
            for j in range(i+1, n):
                dev_genuine.append((tmpls[i][1], tmpls[j][1]))
                if len(dev_genuine) == 50:
                    break
            if len(dev_genuine) == 50:
                break
        if len(dev_genuine) == 50:
            break

    dev_impostor = []
    sids = list(dev_cache.keys())
    for i in range(len(sids)):
        for j in range(i+1, len(sids)):
            dev_impostor.append((dev_cache[sids[i]][0][1], dev_cache[sids[j]][0][1]))
            if len(dev_impostor) == 50:
                break
        if len(dev_impostor) == 50:
            break

    print(f"[+] Formed {len(dev_genuine)} genuine and {len(dev_impostor)} impostor development pairs.")

    delta_grid_deg = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    print("\n" + "=" * 95)
    print("      DEVELOPMENT SET PARAMETER TUNING: UNCERTAINTY BOUNDARY DELTA (fp_training)")
    print("=" * 95)

    for cfg_key, cfg_name in [("b1_512", "Configuration B1 (512-bit, 128 cells)"), ("b2_256", "Configuration B2 (256-bit, 64 cells)")]:
        print(f"\n--- {cfg_name} ---")
        print(f"{'Delta (deg)':<12} {'Gen HD (%)':<15} {'Imp HD (%)':<15} {'Separation':<14} {'d-prime':<10} {'ROC-AUC':<10} {'Overlap':<10} {'Rejections'}")
        print("-" * 95)

        for delta_deg in delta_grid_deg:
            delta_rad = np.radians(delta_deg)
            gen_hds = []
            imp_hds = []
            gen_ovps = []
            imp_ovps = []
            rej_g = 0
            rej_i = 0

            bits_per_sec = 32 if cfg_key == "b1_512" else 16

            for p1, p2 in dev_genuine:
                f1 = p1[cfg_key]
                f2 = p2[cfg_key]
                m1 = (f1["cov"] >= 0.7) & (f1["dbound"] >= delta_rad)
                m2 = (f2["cov"] >= 0.7) & (f2["dbound"] >= delta_rad)
                res = aligned_masked_hd(f1["vec"], f2["vec"], m1, m2, bits_per_sec)
                if res is None:
                    rej_g += 1
                else:
                    gen_hds.append(res["norm_hd"] * 100.0)
                    gen_ovps.append(res["overlap"] * 100.0)

            for p1, p2 in dev_impostor:
                f1 = p1[cfg_key]
                f2 = p2[cfg_key]
                m1 = (f1["cov"] >= 0.7) & (f1["dbound"] >= delta_rad)
                m2 = (f2["cov"] >= 0.7) & (f2["dbound"] >= delta_rad)
                res = aligned_masked_hd(f1["vec"], f2["vec"], m1, m2, bits_per_sec)
                if res is None:
                    rej_i += 1
                else:
                    imp_hds.append(res["norm_hd"] * 100.0)
                    imp_ovps.append(res["overlap"] * 100.0)

            if len(gen_hds) > 0 and len(imp_hds) > 0:
                g_mu, g_sig = np.mean(gen_hds), np.std(gen_hds)
                i_mu, i_sig = np.mean(imp_hds), np.std(imp_hds)
                sep = i_mu - g_mu
                dp = abs(i_mu - g_mu) / np.sqrt(0.5 * (g_sig**2 + i_sig**2))
                auc = compute_auc([-x for x in gen_hds], [-x for x in imp_hds])
                ovp = np.mean(gen_ovps)
                rej_str = f"{rej_g} G / {rej_i} I"
                print(f"{delta_deg:<12.1f} {f'{g_mu:.2f} +- {g_sig:.2f}%':<15} {f'{i_mu:.2f} +- {i_sig:.2f}%':<15} {f'{sep:+.2f}%':<14} {dp:<10.4f} {auc:<10.4f} {f'{ovp:.1f}%':<10} {rej_str}")

if __name__ == "__main__":
    main()

