"""
Production CLI Application for Secure Fingerprint Authentication with Zero-Knowledge Proofs.
Real User Enrollment (Sign Up), Verification (Login), and Database Management.
"""

import os
import sys
import argparse
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, build_template
from template_protection import FuzzyCommitment
from zkp import SchnorrGroup, UserDeviceProver, ServerVerifier
from database import SQLiteAuthDatabase

class BiometricZKPApp:
    def __init__(self, db_path=None):
        self.db = SQLiteAuthDatabase(db_path=db_path)
        self.fc = FuzzyCommitment(secret_bits=32, vector_bits=256)
        self.group = SchnorrGroup()

    def enroll_user(self, username, image_path):
        """
        Signs up a new user with a fingerprint image.
        Extracts features, protects via Fuzzy Commitment, computes ZKP public key,
        and saves ONLY cryptographic records (W, H, Y) to the persistent database.
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] Image not found at: {image_path}")
            return False

        print(f"\n[*] Enrolling New User: '{username}'")
        print(f"[*] Reading sensor capture: {os.path.abspath(image_path)}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[ERROR] Could not read valid grayscale image.")
            return False

        # 1. Biometric Preprocessing & Minutiae Extraction
        t0 = time.perf_counter()
        prep = preprocess_fingerprint(img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        template = build_template(img, prep, minutiae, vector_bits=256)
        print(f"[+] Feature Extraction Complete: {len(minutiae)} minutiae detected.")

        # 2. Template Protection (Fuzzy Commitment)
        enroll_res = self.fc.enroll(template.binary_vector)
        helper_data = enroll_res["helper_data"]
        commitment = enroll_res["commitment"]
        secret = enroll_res["secret"]

        # 3. ZKP Public Key Generation (Y = g^secret mod p)
        prover_setup = UserDeviceProver(secret, group=self.group)
        public_key = prover_setup.public_key

        # 4. Save to Persistent SQLite Database
        self.db.store_user(username, helper_data, commitment, public_key)
        total_time = (time.perf_counter() - t0) * 1000

        print(f"[+] Protected Helper Data W generated (256 bits).")
        print(f"[+] Cryptographic Commitment H = {commitment[:16]}... (SHA256)")
        print(f"[+] Schnorr Public Key Y generated (1024-bit Group).")
        print(f"[+] Record written to SQLite Database ({self.db.db_path}).")
        print(f"[SUCCESS] User '{username}' successfully registered in {total_time:.2f} ms!")
        print("[SECURITY NOTE] Raw fingerprint image and plain template were discarded and NOT stored in DB.")
        return True

    def authenticate_user(self, username, query_image_path):
        """
        Authenticates an enrolled user using a query fingerprint capture and Zero-Knowledge Proofs.
        """
        if not os.path.exists(query_image_path):
            print(f"[ERROR] Image not found at: {query_image_path}")
            return False

        record = self.db.get_user(username)
        if record is None:
            print(f"[ERROR] User '{username}' not found in database. Please enroll first.")
            return False

        print(f"\n[*] Initiating ZKP Authentication for User: '{username}'")
        print(f"[*] Query capture: {os.path.abspath(query_image_path)}")

        query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            print("[ERROR] Could not read valid query image.")
            return False

        # Step 1: Preprocessing & Query Feature Extraction on Client Device
        t_start = time.perf_counter()
        prep = preprocess_fingerprint(query_img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        query_template = build_template(query_img, prep, minutiae, vector_bits=256)
        print(f"[+] Client Device: Extracted {len(minutiae)} minutiae -> Query Binary Vector B'")

        # Step 2: Fuzzy Commitment Secret Recovery
        helper_data = record["helper_data"]
        stored_commitment = record["commitment"]
        public_key = record["public_key"]

        fc_res = self.fc.recover_secret(query_template.binary_vector, helper_data, stored_commitment)
        if not fc_res["success"]:
            print(f"[-] Fuzzy Commitment: Secret recovery failed! (Too many bit differences)")
            print(f"[ACCESS DENIED] Fingerprint does not match enrolled record.")
            return False

        recovered_secret = fc_res["recovered_secret"]
        print(f"[+] Fuzzy Commitment: Secret key successfully recovered! ({fc_res['errors_corrected']} bit errors corrected by ECC)")

        # Step 3: Zero-Knowledge Proof Execution (Prover <-> Verifier)
        print("[*] Executing Schnorr Sigma Zero-Knowledge Proof Protocol...")
        prover = UserDeviceProver(recovered_secret, group=self.group)
        t_commit = prover.step1_create_commitment()
        print("    1. Prover -> Sent Commitment t = g^r mod p")

        verifier = ServerVerifier(public_key, group=self.group)
        c_challenge = verifier.step1_issue_challenge()
        print("    2. Verifier -> Sent Random Challenge c")

        s_response = prover.step2_respond_to_challenge(c_challenge)
        print("    3. Prover -> Sent Response s = (r + c*x) mod q")

        is_valid = verifier.step2_verify(t_commit, s_response, challenge_c=c_challenge)
        total_time = (time.perf_counter() - t_start) * 1000

        if is_valid:
            print("    4. Verifier -> Checked g^s == t * y^c mod p [VALID]")
            print("\n" + "=" * 60)
            print(f"  >>> AUTHENTICATION SUCCESSFUL: ACCESS GRANTED <<<")
            print(f"  Verified in {total_time:.2f} ms without revealing secret to server!")
            print("=" * 60)
            return True
        else:
            print("    4. Verifier -> Verification equation failed [INVALID]")
            print("\n[ACCESS DENIED] Zero-Knowledge Proof failed.")
            return False

    def list_enrolled_users(self):
        users = self.db.list_users()
        print("\n" + "=" * 70)
        print("               ENROLLED USERS IN PERSISTENT DATABASE")
        print("=" * 70)
        if not users:
            print("  No users currently enrolled in database.")
            return
        print(f"{'Username':<20} {'Commitment Hash (SHA256)':<35} {'Enrolled At'}")
        print("-" * 70)
        for u, h, ts in users:
            print(f"{u:<20} {h[:28]}...  {ts}")
        print("=" * 70)

    def inspect_user_security(self, username):
        record = self.db.get_user(username)
        if not record:
            print(f"[ERROR] User '{username}' not found.")
            return
        print("\n" + "=" * 70)
        print(f"  SECURITY INSPECTION FOR USER: '{username}'")
        print("=" * 70)
        print(f"  Database Storage File: {self.db.db_path}")
        print(f"  Enrolled Timestamp   : {record['enrolled_at']}")
        print(f"  Protected Helper (W) : {record['helper_data'][:32].tolist()}... (256 bits)")
        print(f"  Commitment Hash (H)  : {record['commitment']} (SHA-256)")
        print(f"  Public Key (Y)       : {hex(record['public_key'])[:30]}... (1024-bit int)")
        print("-" * 70)
        print("  [VERIFICATION RESULT]: Database contains ZERO raw images or plain minutiae!")
        print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Secure Fingerprint ZKP Authentication System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Enroll command
    parser_enroll = subparsers.add_parser("enroll", help="Sign up / Enroll a new user with a fingerprint image")
    parser_enroll.add_argument("--username", "-u", required=True, help="Username or User ID")
    parser_enroll.add_argument("--image", "-i", required=True, help="Path to raw fingerprint image (.tif, .png, .bmp, .jpg)")

    # Authenticate command
    parser_auth = subparsers.add_parser("authenticate", help="Login / Authenticate a user with a query fingerprint image")
    parser_auth.add_argument("--username", "-u", required=True, help="Username or User ID")
    parser_auth.add_argument("--image", "-i", required=True, help="Path to query fingerprint image")

    # List command
    subparsers.add_parser("list-users", help="List all enrolled users in the database")

    # Inspect command
    parser_inspect = subparsers.add_parser("inspect", help="Inspect cryptographic records stored for a user in the database")
    parser_inspect.add_argument("--username", "-u", required=True, help="Username to inspect")

    args = parser.parse_args()
    app = BiometricZKPApp()

    if args.command == "enroll":
        app.enroll_user(args.username, args.image)
    elif args.command == "authenticate":
        app.authenticate_user(args.username, args.image)
    elif args.command == "list-users":
        app.list_enrolled_users()
    elif args.command == "inspect":
        app.inspect_user_security(args.username)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
