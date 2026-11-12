"""
Step 5 Experimental Evaluation: ECC Feasibility & Fuzzy Commitment Recovery
Evaluates the actual secret-recovery performance of Repetition ECC under realistic
biometric intra-class noise and spatial erasures on fixed_pairs_benchmark.json.

Configurations:
- B1-A: 512-bit, 128 cells, Consecutive Topology (N=512, K=32, R=16)
- B1-B: 512-bit, 128 cells, Interleaved Topology   (N=512, K=32, R=16, Pi_512)
- B2-A: 256-bit, 64 cells,  Consecutive Topology (N=256, K=32, R=8)
- B2-B: 256-bit, 64 cells,  Interleaved Topology   (N=256, K=32, R=8,  Pi_256)

Methodological Standards:
1. Erasure-Aware Majority Voting: Unobserved/invalid polar positions (M_s == 0) are treated as erasures.
2. Explicit Tie Classification: Any replica tie (O_j == Z_j) is flagged as an ambiguous decoding failure, NOT coerced to 0.
3. Identity-Consistent Helper Data: Helper records W and 32-bit secrets C are generated ONCE per enrollment template.
4. Pre-Decoding Alignment Search: All 16 cyclic sector alignments are evaluated before ECC majority voting.
5. Three Impostor Outcomes: (a) Output produced, (b) Secret matched, (c) Commitment verified (FAR with 95% CI).
"""

import os
import sys
import json
import hashlib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.evaluate_step4 import extract_step4_all_configs

def make_permutations(seed=20261112):
    rng = np.random.RandomState(seed)
    p512 = rng.permutation(512).astype(np.int32)
    inv512 = np.argsort(p512).astype(np.int32)
    
    rng = np.random.RandomState(seed)
    p256 = rng.permutation(256).astype(np.int32)
    inv256 = np.argsort(p256).astype(np.int32)
    return (p512, inv512), (p256, inv256)

def hash_secret(secret_vector):
    secret_bytes = np.packbits(np.array(secret_vector, dtype=np.uint8)).tobytes()
    return hashlib.sha256(secret_bytes).hexdigest()

def clopper_pearson_ci(k, n, alpha=0.05):
    """Calculates exact Clopper-Pearson 95% confidence interval for binomial proportions."""
    from scipy.stats import beta
    if k == 0:
        low = 0.0
    else:
        low = float(beta.ppf(alpha / 2.0, k, n - k + 1))
    if k == n:
        high = 1.0
    else:
        high = float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return low * 100.0, high * 100.0

def encode_repetition(secret_bits, rep_factor, topology, perm=None):
    if topology == "consecutive":
        return np.repeat(secret_bits, rep_factor)
    elif topology == "interleaved":
        tiled = np.tile(secret_bits, rep_factor)
        return tiled[perm]
    else:
        raise ValueError(f"Unknown topology {topology}")

def decode_erasure_majority(noisy_cw, mask, k_bits, rep_factor, topology, inv=None):
    """
    Decodes noisy codeword under mutual validity mask.
    Returns:
        recovered_bits: list of 0, 1, or None (if tie or complete erasure)
        stats: dict containing erasures, ties, valid_replicas_per_bit
    """
    recovered = []
    n_erasures = 0
    n_ties = 0
    valid_reps_list = []

    if topology == "consecutive":
        for j in range(k_bits):
            b_slice = noisy_cw[j * rep_factor : (j + 1) * rep_factor]
            m_slice = mask[j * rep_factor : (j + 1) * rep_factor]
            val = b_slice[m_slice > 0]
            n_val = len(val)
            valid_reps_list.append(n_val)

            if n_val == 0:
                n_erasures += 1
                recovered.append(None)
            else:
                ones = int(np.sum(val))
                zeros = n_val - ones
                if ones > zeros:
                    recovered.append(1)
                elif zeros > ones:
                    recovered.append(0)
                else:
                    n_ties += 1
                    recovered.append(None)
    elif topology == "interleaved":
        deint_b = noisy_cw[inv]
        deint_m = mask[inv]
        b_mat = deint_b.reshape(rep_factor, k_bits)
        m_mat = deint_m.reshape(rep_factor, k_bits)

        for j in range(k_bits):
            b_col = b_mat[:, j]
            m_col = m_mat[:, j]
            val = b_col[m_col > 0]
            n_val = len(val)
            valid_reps_list.append(n_val)

            if n_val == 0:
                n_erasures += 1
                recovered.append(None)
            else:
                ones = int(np.sum(val))
                zeros = n_val - ones
                if ones > zeros:
                    recovered.append(1)
                elif zeros > ones:
                    recovered.append(0)
                else:
                    n_ties += 1
                    recovered.append(None)

    is_complete = all(b is not None for b in recovered)
    return recovered, is_complete, {
        "n_erasures": n_erasures,
        "n_ties": n_ties,
        "valid_replicas": valid_reps_list,
        "total_erased_replicas": int(k_bits * rep_factor - np.sum(valid_reps_list))
    }

def evaluate_ecc_configuration(benchmark_data, template_cache, cfg_name, feat_key, total_bits, rep_factor, topology, perm, inv):
    k_bits = 32
    bps = 32 if total_bits == 512 else 16
    tau = 0.7

    # 1. Deterministic Enrollment Store (generated ONCE per unique enrollment image)
    enroll_store = {}
    all_enroll_paths = set(p["path_1"] for p in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"])
    
    for path in all_enroll_paths:
        seed = int(hashlib.sha256(path.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        secret = rng.randint(0, 2, k_bits, dtype=np.uint8)
        commitment = hash_secret(secret)
        
        feat = template_cache[path][feat_key]
        b_enroll = feat["vec"]
        v_enroll = (feat["cov"] >= tau).astype(np.uint8)
        
        cw = encode_repetition(secret, rep_factor, topology, perm=perm)
        helper_data = np.bitwise_xor(b_enroll, cw)
        
        enroll_store[path] = {
            "secret": secret,
            "commitment": commitment,
            "helper_data": helper_data,
            "v_enroll": v_enroll
        }

    # 2. Test Verification Function
    def test_pair(p1_path, p2_path):
        enr = enroll_store[p1_path]
        secret = enr["secret"]
        commitment = enr["commitment"]
        w = enr["helper_data"]
        v1 = enr["v_enroll"]

        feat_q = template_cache[p2_path][feat_key]
        b2 = feat_q["vec"]
        v2 = (feat_q["cov"] >= tau).astype(np.uint8)

        pair_verified = False
        verifying_shift = None
        best_failed_bits = 999
        best_ties = 999
        best_erased_reps = 999
        best_overlap = 0.0
        best_recovered = None
        best_is_complete = False

        for s in range(16):
            sh_b2 = np.roll(b2, s * bps)
            sh_v2 = np.roll(v2, s * bps)
            m_s = (v1 > 0) & (sh_v2 > 0)
            overlap = float(np.mean(m_s))

            noisy_cw = np.bitwise_xor(sh_b2, w)
            rec, is_comp, stats = decode_erasure_majority(noisy_cw, m_s.astype(np.uint8), k_bits, rep_factor, topology, inv=inv)

            # Compare against true secret
            failed_bits = sum(1 for j in range(k_bits) if rec[j] != secret[j])
            ties = stats["n_ties"]
            erased_reps = stats["total_erased_replicas"]

            hash_match = False
            if is_comp:
                cand_hash = hash_secret(rec)
                hash_match = (cand_hash == commitment)

            if hash_match:
                pair_verified = True
                verifying_shift = s
                best_failed_bits = 0
                best_ties = ties
                best_erased_reps = erased_reps
                best_overlap = overlap
                best_recovered = rec
                best_is_complete = True
                break  # Successful authentication!

            if failed_bits < best_failed_bits:
                best_failed_bits = failed_bits
                best_ties = ties
                best_erased_reps = erased_reps
                best_overlap = overlap
                best_recovered = rec
                best_is_complete = is_comp
                verifying_shift = s

        # Evaluate three impostor criteria
        cand_hash_best = hash_secret(best_recovered) if best_is_complete else ""
        comm_verified = (cand_hash_best == commitment)
        exact_secret_recovered = (best_failed_bits == 0)

        return {
            "verified": pair_verified,
            "shift": verifying_shift,
            "failed_bits": best_failed_bits,
            "ties": best_ties,
            "erased_replicas": best_erased_reps,
            "overlap": best_overlap,
            "is_complete": best_is_complete,
            "exact_secret_recovered": exact_secret_recovered,
            "commitment_verified": comm_verified
        }

    # 3. Run Genuine Pairs
    gen_results = []
    for item in benchmark_data["genuine_pairs"]:
        res = test_pair(item["path_1"], item["path_2"])
        gen_results.append(res)

    # 4. Run Impostor Pairs
    imp_results = []
    for item in benchmark_data["impostor_pairs"]:
        res = test_pair(item["path_1"], item["path_2"])
        imp_results.append(res)

    # 5. Compute Aggregate Metrics
    gen_recovered_count = sum(1 for r in gen_results if r["verified"])
    gen_recovery_rate = (gen_recovered_count / len(gen_results)) * 100.0
    gen_frr = 100.0 - gen_recovery_rate

    gen_failed_bits = [r["failed_bits"] for r in gen_results]
    gen_ties = [r["ties"] for r in gen_results]
    gen_erased_reps = [r["erased_replicas"] for r in gen_results]
    gen_shifts = [r["shift"] for r in gen_results if r["verified"]]

    imp_complete_count = sum(1 for r in imp_results if r["is_complete"])
    imp_secret_match_count = sum(1 for r in imp_results if r["exact_secret_recovered"])
    imp_verified_count = sum(1 for r in imp_results if r["commitment_verified"])
    imp_far = (imp_verified_count / len(imp_results)) * 100.0
    imp_far_ci = clopper_pearson_ci(imp_verified_count, len(imp_results))

    imp_failed_bits = [r["failed_bits"] for r in imp_results]
    imp_ties = [r["ties"] for r in imp_results]

    # Failure histogram for genuine (0, 1, 2, 3, 4, 5+)
    hist = {str(i): sum(1 for fb in gen_failed_bits if fb == i) for i in range(5)}
    hist["5+"] = sum(1 for fb in gen_failed_bits if fb >= 5)

    return {
        "cfg_name": cfg_name,
        "total_bits": total_bits,
        "rep_factor": rep_factor,
        "topology": topology,
        "genuine_recovery_rate_pct": gen_recovery_rate,
        "genuine_frr_pct": gen_frr,
        "genuine_failed_bits_mean": float(np.mean(gen_failed_bits)),
        "genuine_failed_bits_std": float(np.std(gen_failed_bits)),
        "genuine_ties_mean": float(np.mean(gen_ties)),
        "genuine_ties_std": float(np.std(gen_ties)),
        "genuine_erased_replicas_mean": float(np.mean(gen_erased_reps)),
        "genuine_erased_replicas_std": float(np.std(gen_erased_reps)),
        "impostor_output_complete_pct": (imp_complete_count / len(imp_results)) * 100.0,
        "impostor_secret_match_pct": (imp_secret_match_count / len(imp_results)) * 100.0,
        "impostor_far_pct": imp_far,
        "impostor_far_95_ci": imp_far_ci,
        "impostor_failed_bits_mean": float(np.mean(imp_failed_bits)),
        "impostor_failed_bits_std": float(np.std(imp_failed_bits)),
        "impostor_ties_mean": float(np.mean(imp_ties)),
        "failed_bits_histogram": hist,
        "successful_shifts_distribution": {str(s): gen_shifts.count(s) for s in set(gen_shifts)}
    }

def main():
    benchmark_json_path = os.path.join(BASE_DIR, "data", "fixed_pairs_benchmark.json")
    print(f"[*] Loading locked benchmark: {benchmark_json_path}")
    with open(benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    unique_paths = set()
    for item in benchmark_data["genuine_pairs"] + benchmark_data["impostor_pairs"]:
        unique_paths.add(item["path_1"])
        unique_paths.add(item["path_2"])

    print(f"[*] Extracting templates for {len(unique_paths)} unique benchmark images...")
    template_cache = {}
    for p in unique_paths:
        template_cache[p] = extract_step4_all_configs(p)
    print("[+] All templates extracted.\n")

    (p512, inv512), (p256, inv256) = make_permutations(seed=20261112)

    configs = [
        ("B1-A", "exp_b1", 512, 16, "consecutive", None, None),
        ("B1-B", "exp_b1", 512, 16, "interleaved", p512, inv512),
        ("B2-A", "exp_b2", 256, 8,  "consecutive", None, None),
        ("B2-B", "exp_b2", 256, 8,  "interleaved", p256, inv256),
    ]

    print("=" * 115)
    print("                    STEP 5: ECC FEASIBILITY & FUZZY COMMITMENT EVALUATION")
    print("=" * 115)

    all_results = {}
    for name, fkey, n_bits, r_factor, topol, p, inv in configs:
        print(f"[*] Running {name}: {n_bits}-bit, R={r_factor}, {topol.upper()} Topology...")
        res = evaluate_ecc_configuration(benchmark_data, template_cache, name, fkey, n_bits, r_factor, topol, p, inv)
        all_results[name] = res

    # Print Formatted Comparison Table
    print("\n" + "=" * 125)
    print(f"{'Config':<8} {'Bits':<6} {'Topology':<13} {'Gen Recovery':<14} {'FRR (%)':<10} {'Failed Bits':<14} {'Ties / Query':<14} {'Erased Reps':<14} {'Imp FAR (95% CI)'}")
    print("-" * 125)

    for name in ["B1-A", "B1-B", "B2-A", "B2-B"]:
        r = all_results[name]
        g_rec = f"{r['genuine_recovery_rate_pct']:.1f}% ({int(r['genuine_recovery_rate_pct'])}/100)"
        frr = f"{r['genuine_frr_pct']:.1f}%"
        f_bits = f"{r['genuine_failed_bits_mean']:.2f} +- {r['genuine_failed_bits_std']:.2f}"
        ties = f"{r['genuine_ties_mean']:.2f} +- {r['genuine_ties_std']:.2f}"
        erased = f"{r['genuine_erased_replicas_mean']:.1f} +- {r['genuine_erased_replicas_std']:.1f}"
        far_ci = f"{r['impostor_far_pct']:.1f}% [{r['impostor_far_95_ci'][0]:.2f}%, {r['impostor_far_95_ci'][1]:.2f}%]"
        print(f"{name:<8} {r['total_bits']:<6} {r['topology']:<13} {g_rec:<14} {frr:<10} {f_bits:<14} {ties:<14} {erased:<14} {far_ci}")

    print("=" * 125)

    # Print Failed Bits Histogram for Genuine Queries
    print("\nGenuine Failed Bits Distribution (Queries with exactly k failed bits out of 32):")
    print(f"{'Config':<8} {'0 (Success)':<13} {'1 bit':<10} {'2 bits':<10} {'3 bits':<10} {'4 bits':<10} {'5+ bits':<10}")
    print("-" * 75)
    for name in ["B1-A", "B1-B", "B2-A", "B2-B"]:
        h = all_results[name]["failed_bits_histogram"]
        print(f"{name:<8} {h['0']:<13} {h['1']:<10} {h['2']:<10} {h['3']:<10} {h['4']:<10} {h['5+']:<10}")

    print("=" * 75)

    out_json = os.path.join(BASE_DIR, "results", "step5_ecc_feasibility_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Full Step 5 evaluation results saved to: {out_json}")

if __name__ == "__main__":
    main()

