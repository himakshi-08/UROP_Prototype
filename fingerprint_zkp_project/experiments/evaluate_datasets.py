"""
Train/Test evaluation of the secure fingerprint template protection framework.

Uses the provided fp_training.zip / fp_testing.zip datasets to compute:
  - Genuine scores
  - Impostor scores
  - FAR, FRR, EER (baseline Hamming similarity)
  - Fuzzy-commitment secret recovery (ECC must recover genuines, reject impostors)
  - Protected + Schnorr ZKP accept/reject rates

Thresholds are selected on the training split and frozen for the testing split.
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data import FingerprintDataset, default_split_dirs
from evaluation import compute_eer, compute_far_frr, compute_roc
from features import BiometricTemplate, extract_minutiae, build_template
from matching import FingerprintMatcher
from preprocessing import preprocess_fingerprint
from template_protection import FuzzyCommitment
from zkp import SchnorrGroup, ServerVerifier, UserDeviceProver


def build_templates(dataset, label=""):
    templates = {}
    n_img = sum(len(v) for v in dataset.values())
    done = 0
    t0 = time.perf_counter()
    for subject_id, samples in dataset.items():
        templates[subject_id] = {}
        for sample_id, img in samples.items():
            prep = preprocess_fingerprint(img)
            minutiae = extract_minutiae(
                prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"]
            )
            tmpl = build_template(img, prep, minutiae, vector_bits=256)
            templates[subject_id][sample_id] = tmpl
            done += 1
            if done % 25 == 0 or done == n_img:
                elapsed = time.perf_counter() - t0
                print(f"  [{label}] {done}/{n_img} images  ({elapsed:.1f}s)")
    return templates


def pairwise_scores(templates, matcher=None):
    genuine = []
    impostor = []
    genuine_hd = []
    impostor_hd = []
    subjects = list(templates.keys())

    for s_id in subjects:
        sample_ids = list(templates[s_id].keys())
        for i in range(len(sample_ids)):
            for j in range(i + 1, len(sample_ids)):
                t1 = templates[s_id][sample_ids[i]]
                t2 = templates[s_id][sample_ids[j]]
                sim = BiometricTemplate.hamming_similarity(t1.binary_vector, t2.binary_vector)
                hd = BiometricTemplate.hamming_distance(t1.binary_vector, t2.binary_vector)
                if matcher is not None:
                    sim = matcher.match(t1, t2)["score"]
                genuine.append(float(sim))
                genuine_hd.append(int(hd))

    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            t1 = templates[s1][list(templates[s1].keys())[0]]
            t2 = templates[s2][list(templates[s2].keys())[0]]
            sim = BiometricTemplate.hamming_similarity(t1.binary_vector, t2.binary_vector)
            hd = BiometricTemplate.hamming_distance(t1.binary_vector, t2.binary_vector)
            if matcher is not None:
                sim = matcher.match(t1, t2)["score"]
            impostor.append(float(sim))
            impostor_hd.append(int(hd))

    return {
        "genuine_scores": genuine,
        "impostor_scores": impostor,
        "genuine_hd": genuine_hd,
        "impostor_hd": impostor_hd,
    }


def fuzzy_recovery_stats(templates, secret_bits=64):
    fc = FuzzyCommitment(secret_bits=secret_bits, vector_bits=256)
    enrolled = {}
    for s_id, samples in templates.items():
        first = list(samples.keys())[0]
        enrolled[s_id] = fc.enroll(samples[first].binary_vector)

    genuine_ok = 0
    genuine_n = 0
    impostor_ok = 0
    impostor_n = 0
    genuine_scores = []
    impostor_scores = []

    for s_id, samples in templates.items():
        helper = enrolled[s_id]["helper_data"]
        commitment = enrolled[s_id]["commitment"]
        sample_ids = list(samples.keys())
        enroll_vec = samples[sample_ids[0]].binary_vector
        for sample_id in sample_ids[1:]:
            query = samples[sample_id]
            res = fc.recover_secret(query.binary_vector, helper, commitment)
            genuine_n += 1
            genuine_ok += int(bool(res["success"]))
            genuine_scores.append(
                float(BiometricTemplate.hamming_similarity(enroll_vec, query.binary_vector))
            )

    subjects = list(templates.keys())
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            helper = enrolled[s1]["helper_data"]
            commitment = enrolled[s1]["commitment"]
            t1 = templates[s1][list(templates[s1].keys())[0]]
            t2 = templates[s2][list(templates[s2].keys())[0]]
            res = fc.recover_secret(t2.binary_vector, helper, commitment)
            impostor_n += 1
            impostor_ok += int(bool(res["success"]))
            impostor_scores.append(
                float(BiometricTemplate.hamming_similarity(t1.binary_vector, t2.binary_vector))
            )

    return {
        "genuine_recovery": genuine_ok,
        "genuine_attempts": genuine_n,
        "impostor_recovery": impostor_ok,
        "impostor_attempts": impostor_n,
        "genuine_recovery_rate": genuine_ok / max(1, genuine_n),
        "impostor_recovery_rate": impostor_ok / max(1, impostor_n),
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "enrolled": enrolled,
    }


def zkp_stats(templates, fc_stats):
    group = SchnorrGroup()
    genuine_ok = 0
    genuine_n = 0
    impostor_ok = 0
    impostor_n = 0
    proof_ms = []
    verify_ms = []

    enrolled = fc_stats["enrolled"]
    fc = FuzzyCommitment(secret_bits=64, vector_bits=256)

    for s_id, samples in templates.items():
        helper = enrolled[s_id]["helper_data"]
        commitment = enrolled[s_id]["commitment"]
        secret = enrolled[s_id]["secret"]
        prover0 = UserDeviceProver(secret, group=group)
        y = prover0.public_key
        sample_ids = list(samples.keys())
        for sample_id in sample_ids[1:]:
            genuine_n += 1
            rec = fc.recover_secret(samples[sample_id].binary_vector, helper, commitment)
            if not rec["success"]:
                continue
            t0 = time.perf_counter()
            prover = UserDeviceProver(rec["recovered_secret"], group=group)
            t_commit = prover.step1_create_commitment()
            verifier = ServerVerifier(y, group=group)
            chal = verifier.step1_issue_challenge()
            s_resp = prover.step2_respond_to_challenge(chal)
            t_proof = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            ok = verifier.step2_verify(t_commit, s_resp, challenge_c=chal)
            t_ver = (time.perf_counter() - t1) * 1000
            proof_ms.append(t_proof)
            verify_ms.append(t_ver)
            genuine_ok += int(bool(ok))

    subjects = list(templates.keys())
    for i in range(len(subjects)):
        for j in range(i + 1, len(subjects)):
            s1, s2 = subjects[i], subjects[j]
            helper = enrolled[s1]["helper_data"]
            commitment = enrolled[s1]["commitment"]
            secret = enrolled[s1]["secret"]
            y = UserDeviceProver(secret, group=group).public_key
            query = templates[s2][list(templates[s2].keys())[0]]
            impostor_n += 1
            rec = fc.recover_secret(query.binary_vector, helper, commitment)
            if rec["success"]:
                prover = UserDeviceProver(rec["recovered_secret"], group=group)
                t_commit = prover.step1_create_commitment()
                verifier = ServerVerifier(y, group=group)
                chal = verifier.step1_issue_challenge()
                s_resp = prover.step2_respond_to_challenge(chal)
                ok = verifier.step2_verify(t_commit, s_resp, challenge_c=chal)
                impostor_ok += int(bool(ok))

    far = impostor_ok / max(1, impostor_n)
    frr = 1.0 - (genuine_ok / max(1, genuine_n))
    return {
        "genuine_accepts": genuine_ok,
        "genuine_attempts": genuine_n,
        "impostor_accepts": impostor_ok,
        "impostor_attempts": impostor_n,
        "far": far,
        "frr": frr,
        "eer": 0.5 * (far + frr),
        "avg_proof_ms": float(np.mean(proof_ms)) if proof_ms else 0.0,
        "avg_verify_ms": float(np.mean(verify_ms)) if verify_ms else 0.0,
    }


def summarize_split(name, templates, matcher):
    print(f"\n=== {name.upper()} pairwise scores ===")
    scores = pairwise_scores(templates, matcher=matcher)
    g = np.array(scores["genuine_scores"], dtype=np.float64)
    i = np.array(scores["impostor_scores"], dtype=np.float64)
    eer = compute_eer(g, i)
    print(
        f"  Genuine pairs={len(g)}  mean={g.mean():.4f}  std={g.std():.4f}  "
        f"HD mean={np.mean(scores['genuine_hd']):.1f}"
    )
    print(
        f"  Impostor pairs={len(i)}  mean={i.mean():.4f}  std={i.std():.4f}  "
        f"HD mean={np.mean(scores['impostor_hd']):.1f}"
    )
    print(
        f"  EER={eer['eer']*100:.2f}%  FAR={eer['far_at_eer']*100:.2f}%  "
        f"FRR={eer['frr_at_eer']*100:.2f}%  tau={eer['threshold']:.4f}"
    )

    print(f"\n=== {name.upper()} fuzzy commitment / ECC ===")
    fc = fuzzy_recovery_stats(templates)
    print(
        f"  Genuine recovery {fc['genuine_recovery']}/{fc['genuine_attempts']} "
        f"({fc['genuine_recovery_rate']*100:.2f}%)"
    )
    print(
        f"  Impostor recovery {fc['impostor_recovery']}/{fc['impostor_attempts']} "
        f"({fc['impostor_recovery_rate']*100:.2f}%)  [should be 0 — no false accept]"
    )

    print(f"\n=== {name.upper()} protected + ZKP ===")
    zkp = zkp_stats(templates, fc)
    print(
        f"  FAR={zkp['far']*100:.2f}%  FRR={zkp['frr']*100:.2f}%  "
        f"EER={zkp['eer']*100:.2f}%  proof={zkp['avg_proof_ms']:.2f} ms"
    )
    return scores, eer, fc, zkp


def save_artifacts(test_scores, test_eer, train_eer, test_fc, test_zkp, train_fc):
    fig_dir = os.path.join(BASE_DIR, "results", "figures")
    table_dir = os.path.join(BASE_DIR, "results", "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    g = test_scores["genuine_scores"]
    i = test_scores["impostor_scores"]
    tau = test_eer["threshold"]

    plt.figure(figsize=(8, 5))
    plt.hist(g, bins=24, alpha=0.65, label="Genuine scores", color="#00c853", density=True)
    plt.hist(i, bins=24, alpha=0.65, label="Impostor scores", color="#ff1744", density=True)
    plt.axvline(tau, color="black", linestyle="--", label=f"EER threshold ({tau:.2f})")
    plt.title("Test Set: Genuine vs Impostor Scores")
    plt.xlabel("Similarity score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "baseline_score_distribution.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(test_eer["thresholds"], test_eer["far_curve"], label="FAR", color="#ff1744")
    plt.plot(test_eer["thresholds"], test_eer["frr_curve"], label="FRR", color="#2979ff")
    plt.axvline(tau, color="black", linestyle="--", label=f"EER = {test_eer['eer']*100:.2f}%")
    plt.title("Test Set: FAR and FRR")
    plt.xlabel("Threshold")
    plt.ylabel("Error rate")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "baseline_far_frr_curve.png"), dpi=180)
    plt.close()

    roc = compute_roc(test_eer["far_curve"], 1.0 - test_eer["frr_curve"])
    plt.figure(figsize=(7, 6))
    plt.plot(roc["far"], roc["tar"], color="#7c4dff", lw=2, label=f"ROC (AUC={roc['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("Test Set: ROC Curve")
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "baseline_roc_curve.png"), dpi=180)
    plt.close()

    rows = [
        {
            "system": "Baseline (binary template)",
            "split": "test",
            "genuine_score_mean": float(np.mean(g)),
            "impostor_score_mean": float(np.mean(i)),
            "genuine_pairs": len(g),
            "impostor_pairs": len(i),
            "eer": float(test_eer["eer"]),
            "far": float(test_eer["far_at_eer"]),
            "frr": float(test_eer["frr_at_eer"]),
            "threshold": float(test_eer["threshold"]),
            "template_protected": False,
            "zkp": False,
            "proof_time_ms": None,
        },
        {
            "system": "Protected (Fuzzy Commitment + ECC)",
            "split": "test",
            "genuine_score_mean": float(np.mean(test_fc["genuine_scores"])) if test_fc["genuine_scores"] else 0.0,
            "impostor_score_mean": float(np.mean(test_fc["impostor_scores"])) if test_fc["impostor_scores"] else 0.0,
            "genuine_pairs": test_fc["genuine_attempts"],
            "impostor_pairs": test_fc["impostor_attempts"],
            "eer": float(0.5 * (test_fc["impostor_recovery_rate"] + (1.0 - test_fc["genuine_recovery_rate"]))),
            "far": float(test_fc["impostor_recovery_rate"]),
            "frr": float(1.0 - test_fc["genuine_recovery_rate"]),
            "threshold": "ECC majority vote",
            "template_protected": True,
            "zkp": False,
            "proof_time_ms": None,
        },
        {
            "system": "Protected + Schnorr ZKP",
            "split": "test",
            "genuine_score_mean": float(np.mean(test_fc["genuine_scores"])) if test_fc["genuine_scores"] else 0.0,
            "impostor_score_mean": float(np.mean(test_fc["impostor_scores"])) if test_fc["impostor_scores"] else 0.0,
            "genuine_pairs": test_zkp["genuine_attempts"],
            "impostor_pairs": test_zkp["impostor_attempts"],
            "eer": float(test_zkp["eer"]),
            "far": float(test_zkp["far"]),
            "frr": float(test_zkp["frr"]),
            "threshold": "ECC + ZKP verify",
            "template_protected": True,
            "zkp": True,
            "proof_time_ms": float(test_zkp["avg_proof_ms"]),
        },
    ]

    payload = {
        "train_threshold": float(train_eer["threshold"]),
        "train_eer": float(train_eer["eer"]),
        "train_fc": {
            "genuine_recovery_rate": train_fc["genuine_recovery_rate"],
            "impostor_recovery_rate": train_fc["impostor_recovery_rate"],
        },
        "table": rows,
        "test_score_summary": {
            "genuine_mean": float(np.mean(g)),
            "genuine_std": float(np.std(g)),
            "impostor_mean": float(np.mean(i)),
            "impostor_std": float(np.std(i)),
            "genuine_hd_mean": float(np.mean(test_scores["genuine_hd"])),
            "impostor_hd_mean": float(np.mean(test_scores["impostor_hd"])),
        },
    }

    json_path = os.path.join(table_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(table_dir, "final_comparison_table.txt")
    hdr = f"{'System':<36} {'Genuine':>10} {'Impostor':>10} {'FAR':>10} {'FRR':>10} {'EER':>10}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['system']:<36} {r['genuine_score_mean']:>10.4f} {r['impostor_score_mean']:>10.4f} "
            f"{r['far']*100:>9.2f}% {r['frr']*100:>9.2f}% {r['eer']*100:>9.2f}%"
        )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved {json_path}")
    print(f"Saved {txt_path}")
    return payload


def run_evaluation(max_train_subjects=40, max_test_subjects=40):
    dirs = default_split_dirs()
    print("=" * 72)
    print("  DATASET EVALUATION: fp_training.zip  /  fp_testing.zip")
    print("=" * 72)

    train_ds = FingerprintDataset(dirs["train"]).load_dataset(max_subjects=max_train_subjects)
    test_ds = FingerprintDataset(dirs["test"]).load_dataset(max_subjects=max_test_subjects)

    print("\n[1] Building training templates...")
    train_tmpl = build_templates(train_ds, label="train")
    print("\n[2] Building testing templates...")
    test_tmpl = build_templates(test_ds, label="test")

    matcher = FingerprintMatcher(mode="binary", default_threshold=0.5)
    train_scores, train_eer, train_fc, _ = summarize_split("train", train_tmpl, matcher)
    test_scores, test_eer, test_fc, test_zkp = summarize_split("test", test_tmpl, matcher)

    # Apply training EER threshold to the test scores (held-out operating point).
    far_frr = compute_far_frr(
        test_scores["genuine_scores"],
        test_scores["impostor_scores"],
        thresholds=np.array([train_eer["threshold"]]),
    )
    print("\n[Held-out] Test FAR/FRR at training EER threshold "
          f"{train_eer['threshold']:.4f}: FAR={far_frr['far'][0]*100:.2f}%  "
          f"FRR={far_frr['frr'][0]*100:.2f}%")

    return save_artifacts(test_scores, test_eer, train_eer, test_fc, test_zkp, train_fc)


if __name__ == "__main__":
    run_evaluation()
