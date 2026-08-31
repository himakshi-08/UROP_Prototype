# Secure Fingerprint Template Protection System Using Zero-Knowledge Proofs

**UROP Research Prototype**  
Privacy-Preserving Biometric Authentication via Fuzzy Commitment and Schnorr Sigma Protocols.

---

## 🔬 How It Works

The system is built on four security layers that work together to authenticate a user **without ever exposing their biometric data**:

```
[Step 1: Sign Up / Enroll]
  User provides fingerprint image
        │
        ▼
  Preprocessing (Normalize → Gabor Filter → Skeletonize)
        │
        ▼
  Minutiae Extraction (x, y, orientation, type)
        │
        ▼
  256-bit Binary Template B created
        │
        ▼
  Random Secret C generated (CSPRNG, 64 bits)
  ECC Encodes C into Codeword (256 bits)
  Helper Data W = B XOR Codeword(C)
  Commitment H = SHA256(C)
  Public Key Y = g^C mod p (Schnorr Group)
        │
        ▼
  ONLY (W, H, Y) saved to SQLite Database
  Raw image and template DISCARDED

[Step 2: Login / Authenticate]
  User presents fingerprint image again
        │
        ▼
  Same preprocessing + minutiae extraction
  Query Binary Template B' created
        │
        ▼
  Fuzzy Commitment Recovery:
    Noisy Codeword = B' XOR W
    ECC Error-Corrects to recover candidate C'
    Checks: SHA256(C') == H  →  Yes/No?
        │
        ▼
  If YES → Schnorr Zero-Knowledge Proof (3-pass)
    1. Prover sends Commitment t = g^r mod p
    2. Verifier sends random Challenge c
    3. Prover sends Response s = (r + c * x) mod q
    4. Verifier checks: g^s == t * y^c mod p
        │
        ▼
  If equation holds → ACCESS GRANTED
  If any step fails → ACCESS DENIED
```

### Security Guarantee
- The **server never sees** the fingerprint image, minutiae, or secret key.
- The database stores only **cryptographic noise** — even if stolen, an attacker cannot reconstruct the biometric.
- Each authentication uses a **fresh random challenge** — replay attacks are impossible.
- Templates are **revocable** — if compromised, the user re-enrolls with new helper data W.

---

## 📁 Project Structure

```
fingerprint_zkp_project/
├── app.py                  # Main CLI application (enroll, authenticate, inspect, list-users)
├── data/
│   ├── raw/                # Raw fingerprint images (FVC2002 DB1_B .tif format)
│   └── dataset_loader.py   # Loads and auto-generates dataset
├── preprocessing/
│   ├── normalization.py    # Mean & variance image normalization
│   ├── enhancement.py      # Orientation estimation & Gabor wavelet filtering
│   ├── segmentation.py     # Block std-deviation foreground ROI segmentation
│   ├── thinning.py         # Adaptive binarization & skeletonization
│   └── pipeline.py         # Unified preprocessing orchestration
├── features/
│   ├── minutiae.py         # Crossing Number minutiae detector (endings & bifurcations)
│   └── template.py         # 256-bit error-tolerant spatial grid binary template B
├── matching/
│   ├── matcher.py          # Minutiae alignment & hybrid similarity matcher
│   └── distance.py         # Hamming & Euclidean distance metrics
├── evaluation/
│   ├── far_frr.py          # FAR and FRR computation
│   ├── roc.py              # ROC curve generator & AUC
│   └── eer.py              # Equal Error Rate (EER) determination
├── template_protection/
│   ├── key_generation.py   # CSPRNG cryptographic secret key generation (64-bit)
│   ├── error_correction.py # Majority-voting block ECC (64-bit → 256-bit codeword)
│   └── fuzzy_commitment.py # Enrollment W computation & noisy secret recovery
├── zkp/
│   ├── schnorr.py          # Schnorr Sigma Protocol (RFC 2409 1024-bit Safe Prime)
│   ├── prover.py           # User device prover interface
│   └── verifier.py         # Server verifier interface
├── authentication/
│   └── authentication_pipeline.py  # End-to-end orchestration
├── database/
│   ├── db_manager.py       # SQLite persistent database manager
│   └── auth_system.db      # Persistent database (only W, H, Y stored — no images)
├── experiments/
│   ├── baseline.py         # Experiment 1 & 2: Baseline biometric EER evaluation
│   ├── protected.py        # Experiment 3: Fuzzy commitment template protection
│   ├── zkp_experiment.py   # Experiment 4: Full system benchmark comparison
│   └── visualize_results.py# Generates presentation figures
├── results/
│   ├── figures/            # ROC curves, score distributions, pipeline charts
│   └── tables/             # Final comparison benchmark tables
├── tests/                  # Automated pytest unit test suite (12 tests)
└── requirements.txt
```

---

## 🚀 Quickstart

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📲 Running the Application (`app.py`)

### 1. Enroll / Sign Up a New User

**Command:**
```bash
python app.py enroll --username alice --image data/raw/101_1.tif
```

**Actual Output:**
```
[*] Enrolling New User: 'alice'
[*] Reading sensor capture: data/raw/101_1.tif
[+] Feature Extraction Complete: 63 minutiae detected.
[+] Protected Helper Data W generated (256 bits).
[+] Cryptographic Commitment H = 764b031ad39bb831... (SHA256)
[+] Schnorr Public Key Y generated (1024-bit Group).
[+] Record written to SQLite Database (database/auth_system.db).
[SUCCESS] User 'alice' successfully registered in 207.33 ms!
[SECURITY NOTE] Raw fingerprint image and plain template were discarded and NOT stored in DB.
```

---

### 2. Authenticate / Login an Enrolled User

**Command:**
```bash
python app.py authenticate --username alice --image data/raw/101_1.tif
```

**Actual Output:**
```
[*] Initiating ZKP Authentication for User: 'alice'
[*] Query capture: data/raw/101_1.tif
[+] Client Device: Extracted 63 minutiae -> Query Binary Vector B'
[+] Fuzzy Commitment: Secret key successfully recovered! (0 bit errors corrected by ECC)
[*] Executing Schnorr Sigma Zero-Knowledge Proof Protocol...
    1. Prover -> Sent Commitment t = g^r mod p
    2. Verifier -> Sent Random Challenge c
    3. Prover -> Sent Response s = (r + c*x) mod q
    4. Verifier -> Checked g^s == t * y^c mod p [VALID]

============================================================
  >>> AUTHENTICATION SUCCESSFUL: ACCESS GRANTED <<<
  Verified in 217.26 ms without revealing secret to server!
============================================================
```

---

### 3. Impostor Attack (Access Denied)

An attacker claiming to be `alice` with someone else's fingerprint:

**Command:**
```bash
python app.py authenticate --username alice --image data/raw/102_1.tif
```

**Actual Output:**
```
[*] Initiating ZKP Authentication for User: 'alice'
[*] Query capture: data/raw/102_1.tif
[+] Client Device: Extracted 78 minutiae -> Query Binary Vector B'
[-] Fuzzy Commitment: Secret recovery failed! (Too many bit differences)
[ACCESS DENIED] Fingerprint does not match enrolled record.
```

---

### 4. Inspect What is Stored in the Database

**Command:**
```bash
python app.py inspect --username alice
```

**Actual Output:**
```
======================================================================
  SECURITY INSPECTION FOR USER: 'alice'
======================================================================
  Database Storage File: database/auth_system.db
  Enrolled Timestamp   : 2026-08-31 13:22:27
  Protected Helper (W) : [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, ...]... (256 bits)
  Commitment Hash (H)  : 764b031ad39bb831eceeede4329477af577446578586ef00d139fad03909cc73 (SHA-256)
  Public Key (Y)       : 0x29741c8d48059e947a5938ebab12... (1024-bit int)
----------------------------------------------------------------------
  [VERIFICATION RESULT]: Database contains ZERO raw images or plain minutiae!
======================================================================
```

---

### 5. List All Enrolled Users

**Command:**
```bash
python app.py list-users
```

**Actual Output:**
```
======================================================================
               ENROLLED USERS IN PERSISTENT DATABASE
======================================================================
Username             Commitment Hash (SHA256)            Enrolled At
----------------------------------------------------------------------
user_101             9d5cd9b88ae0fd3e74fb5b5b0cab...  2026-08-31 13:16:11
alice                764b031ad39bb831eceeede43294...  2026-08-31 13:22:27
======================================================================
```

---

## 🧪 Running Research Experiments

### Automated Tests (12 Tests, All Passing)
```bash
python -m pytest tests/
```

**Output:**
```
tests/test_preprocessing.py   .....  [5 passed]
tests/test_features.py        ..     [2 passed]
tests/test_fuzzy_commitment.py ...   [3 passed]
tests/test_zkp.py             .      [1 passed]
tests/test_pipeline.py        .      [1 passed]
============================= 12 passed in 1.30s ==============================
```

### Benchmark Experiments

```bash
python experiments/baseline.py          # Baseline biometric EER
python experiments/protected.py         # Fuzzy Commitment secret recovery rates
python experiments/zkp_experiment.py    # Full system comparison table
python experiments/visualize_results.py # Generate all visual figures
```

---

## 📊 Benchmark Results (On FVC2002 DB1_B Dataset)

| System | EER | FAR | FRR | Template Protected | ZKP | Proof Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 35.10% | 35.56% | 34.64% | No | No | — |
| **Protected** (Fuzzy Commitment) | 37.06% | 0.00% | 95.71% | Yes | No | — |
| **Protected + ZKP** | 49.68% | 2.22% | 97.14% | Yes | Yes | **7.05 ms** |

### Cryptographic Overhead Metrics

| Metric | Value |
| :--- | :--- |
| Proof Generation Time | 7.05 ms |
| Proof Verification Time | 14.42 ms |
| Authentication Latency | ~419 ms (mostly preprocessing) |
| ZKP Proof Size | 288 bytes |
| DB Storage Overhead / User | 320 bytes (W + H + Y) |
| Communication Overhead / Auth | 320 bytes |

---

## 🗂️ Adding Your Own Dataset

Place fingerprint images in `data/raw/` with the naming convention:
```
{subject_id}_{sample_id}.tif
e.g. 101_1.tif, 101_2.tif, 102_1.tif ...
```

Recommended: FVC2000 / FVC2002 / FVC2004 (available via University of Bologna FVC page or Kaggle).
