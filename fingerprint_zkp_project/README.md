# Secure Fingerprint Template Protection System Using Zero-Knowledge Proofs

**UROP Research Prototype**  
Privacy-Preserving Biometric Authentication via Fuzzy Commitment and Schnorr Sigma Protocols.

---

## 🔬 Core Architecture (4 Layers)

```
[Fingerprint Image]
       │
       ▼
[Layer 1: Preprocessing & Minutiae Extraction] (Normalization, Gabor Filter, Skeletonization, Crossing Number)
       │
       ▼
[Layer 2: Biometric Matching & Baseline EER] (Alignment & Hamming Distance Similarity)
       │
       ▼
[Layer 3: Biometric Template Protection] (Fuzzy Commitment: W = B XOR Codeword(C), H = SHA256(C))
       │
       ▼
[Layer 4: Zero-Knowledge Proof Authentication] (Schnorr Sigma Protocol: t = g^r mod p, s = r + c*x mod q)
       │
       ▼
[Authentication Decision: ACCEPT / REJECT]
```

---

## 📁 Project Structure

```
fingerprint_zkp_project/
├── data/
│   ├── raw/                # Raw fingerprint images (e.g., FVC2000/2002/2004 or synthetic)
│   └── processed/          # Preprocessed skeleton & minutiae cache
├── preprocessing/
│   ├── normalization.py    # Mean & variance image normalization
│   ├── enhancement.py      # Orientation estimation & Gabor wavelet enhancement
│   ├── segmentation.py     # Block standard-deviation foreground ROI segmentation
│   ├── thinning.py         # Adaptive binarization & Zhang-Suen skeletonization
│   └── pipeline.py         # Unified preprocessing orchestration
├── features/
│   ├── minutiae.py         # Crossing number minutiae detector (endings & bifurcations)
│   └── template.py         # 256-bit error-tolerant spatial grid binary template B
├── matching/
│   ├── matcher.py          # Minutiae alignment & hybrid similarity matcher
│   └── distance.py         # Spatial minutiae & Hamming distance metrics
├── evaluation/
│   ├── far_frr.py          # FAR and FRR computation across threshold sweeps
│   ├── roc.py              # ROC curve generator & AUC calculation
│   └── eer.py              # Equal Error Rate (EER) determination
├── template_protection/
│   ├── key_generation.py   # CSPRNG cryptographic secret key generation (64-bit)
│   ├── error_correction.py # Majority-voting block ECC (64-bit secret -> 256-bit codeword)
│   └── fuzzy_commitment.py # Helper data W computation & noisy secret recovery
├── zkp/
│   ├── schnorr.py          # Schnorr Sigma Protocol over RFC 2409 1024-bit Safe Prime Group
│   ├── prover.py           # User device prover interface
│   └── verifier.py         # Server verifier interface
├── authentication/
│   └── authentication_pipeline.py # End-to-end enrollment & authentication database
├── experiments/
│   ├── baseline.py         # Experiment 1 & 2: Baseline biometric evaluation
│   ├── protected.py        # Experiment 3: Fuzzy commitment template protection
│   ├── zkp_experiment.py   # Experiment 4: Full system integration & comparison table
│   └── visualize_results.py# Generates presentation figures for UROP faculty
├── results/
│   ├── figures/            # Saved ROC curves, score distributions, and pipeline charts
│   └── tables/             # Final comparison benchmark tables
├── tests/                  # Automated pytest unit test suite (12 tests across all layers)
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests/
```

### 3. Run Experiments & Benchmark Comparisons
* **Baseline Biometric System**:
  ```bash
  python experiments/baseline.py
  ```
* **Biometric Template Protection (Fuzzy Commitment)**:
  ```bash
  python experiments/protected.py
  ```
* **Full Integrated System (Baseline vs. Protected vs. Protected+ZKP)**:
  ```bash
  python experiments/zkp_experiment.py
  ```
* **Generate Faculty Demonstration Figures**:
  ```bash
  python experiments/visualize_results.py
  ```

---

## 📊 Evaluation & Visual Deliverables

All generated figures and tables are stored in `results/`:
* `results/figures/01_preprocessing_and_minutiae_stages.png`: Full 5-stage image pipeline visualization.
* `results/figures/02_fuzzy_commitment_binary_grid.png`: 256-bit error-tolerant binary template representation.
* `results/figures/03_authentication_timing_breakdown.png`: Millisecond latency benchmark per layer.
* `results/figures/baseline_roc_curve.png`: Baseline ROC curve and AUC.
* `results/figures/baseline_far_frr_curve.png`: FAR vs FRR curves with marked EER point.
* `results/figures/baseline_score_distribution.png`: Genuine vs Impostor similarity distribution.
* `results/tables/final_comparison_table.txt`: Full research benchmark table.
