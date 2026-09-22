"""
Interactive Web Application Server for UROP Biometric ZKP Prototype.
Provides a modern visual interface for:
1. Live Fingerprint Preprocessing & Minutiae Visualization
2. Fuzzy Commitment Cryptographic Enrollment
3. Interactive 3-Pass Schnorr Sigma Zero-Knowledge Proof Authentication
4. Database & Privacy Auditing (verifying zero raw template leakage)
5. Research Evaluation Figures & Benchmarks
"""

import os
import sys
import json
import base64
import time
import re
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import cv2
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import preprocess_fingerprint
from features import extract_minutiae, BiometricTemplate, build_template
from template_protection import FuzzyCommitment
from zkp import SchnorrGroup, UserDeviceProver, ServerVerifier
from database import SQLiteAuthDatabase

class BiometricBackend:
    def __init__(self):
        self.db_path = os.path.join(BASE_DIR, "database", "auth_system.db")
        self.db = SQLiteAuthDatabase(db_path=self.db_path)
        self.fc = FuzzyCommitment(secret_bits=32, vector_bits=256)
        self.group = SchnorrGroup()
        self.raw_data_dir = os.path.join(BASE_DIR, "data", "raw")
        self._ensure_default_dataset_users()

    def _ensure_default_dataset_users(self):
        defaults = [
            ("user_000", "000_L0_0.bmp"),
            ("user_001", "001_L0_0.bmp"),
            ("user_002", "002_L0_0.bmp"),
            ("alice", "000_L0_0.bmp"),
        ]
        for uname, fname in defaults:
            if not self.db.get_user(uname):
                img, err = self.load_image(fname)
                if img is not None:
                    self.enroll_user(uname, fname)

    def get_samples(self):
        pattern = re.compile(r'(\d+)(?:_L\d+)?_(\d+)\.(tif|tiff|png|jpg|bmp)', re.IGNORECASE)
        samples = []
        dirs = [
            os.path.join(BASE_DIR, "data", "fp_testing"),
            os.path.join(BASE_DIR, "data", "fp_training"),
            os.path.join(BASE_DIR, "data", "raw"),
        ]
        for d in dirs:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                for fname in sorted(files):
                    m = pattern.match(fname)
                    if m:
                        rel = os.path.relpath(os.path.join(root, fname), BASE_DIR).replace("\\", "/")
                        samples.append({
                            "filename": fname,
                            "subject_id": int(m.group(1)),
                            "sample_id": int(m.group(2)),
                            "path": rel
                        })
                        if len(samples) >= 120:
                            break
                if len(samples) >= 120:
                    break
            if len(samples) >= 120:
                break
        return samples

    def load_image(self, file_param):
        if not file_param:
            return None, "Empty file parameter"

        path = None
        if os.path.isabs(file_param) and os.path.exists(file_param):
            path = file_param
        else:
            # 1. Try exact relative path from BASE_DIR
            candidate1 = os.path.join(BASE_DIR, file_param)
            if os.path.exists(candidate1):
                path = candidate1
            else:
                # 2. Try raw_data_dir with basename
                fname = os.path.basename(file_param)
                candidate2 = os.path.join(self.raw_data_dir, fname)
                if os.path.exists(candidate2):
                    path = candidate2
                else:
                    # 3. Recursive search in data/ directory
                    data_dir = os.path.join(BASE_DIR, "data")
                    if os.path.exists(data_dir):
                        for root, _, files in os.walk(data_dir):
                            if fname in files:
                                path = os.path.join(root, fname)
                                break

        if not path or not os.path.exists(path):
            return None, f"File not found: {file_param}"

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, f"Failed to decode image from {path}"
        return img, None

    def analyze_fingerprint(self, img):
        t0 = time.perf_counter()
        prep = preprocess_fingerprint(img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        template = build_template(img, prep, minutiae, vector_bits=256)
        t_elapsed = (time.perf_counter() - t0) * 1000

        # Create minutiae overlay
        skel_norm = (prep["skeleton"] * 255).astype(np.uint8)
        color_overlay = cv2.cvtColor(skel_norm, cv2.COLOR_GRAY2BGR)
        
        endings = 0
        bifurcations = 0
        minutiae_coords = []

        for m in minutiae:
            if m.type == "ending":
                endings += 1
                color = (0, 0, 255)  # Red for ridge endings
            else:
                bifurcations += 1
                color = (255, 200, 0)  # Cyan/Yellow for bifurcations

            cv2.circle(color_overlay, (m.x, m.y), 3, color, 1)
            # Small orientation ray
            dx = int(7 * math.cos(m.orientation))
            dy = int(7 * math.sin(m.orientation))
            cv2.line(color_overlay, (m.x, m.y), (m.x + dx, m.y + dy), color, 1)

            if len(minutiae_coords) < 30:
                minutiae_coords.append({
                    "x": m.x,
                    "y": m.y,
                    "theta": round(float(m.orientation), 3),
                    "type": m.type
                })

        def to_b64(mat):
            _, buf = cv2.imencode('.png', mat)
            return base64.b64encode(buf).decode('utf-8')

        return {
            "elapsed_ms": round(t_elapsed, 2),
            "minutiae_count": len(minutiae),
            "endings_count": endings,
            "bifurcations_count": bifurcations,
            "minutiae_samples": minutiae_coords,
            "binary_vector_preview": template.binary_vector[:64].tolist(),
            "binary_vector_ones": int(np.sum(template.binary_vector)),
            "images": {
                "raw": to_b64(img),
                "normalized": to_b64(prep["normalized"]),
                "enhanced": to_b64(prep["enhanced"]),
                "skeleton": to_b64(skel_norm),
                "minutiae_overlay": to_b64(color_overlay)
            }
        }

    def enroll_user(self, username, file_param):
        img, err = self.load_image(file_param)
        if err:
            return {"success": False, "error": err}

        t0 = time.perf_counter()
        prep = preprocess_fingerprint(img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        template = build_template(img, prep, minutiae, vector_bits=256)

        enroll_res = self.fc.enroll(template.binary_vector)
        helper_data = enroll_res["helper_data"]
        commitment = enroll_res["commitment"]
        secret = enroll_res["secret"]

        prover_setup = UserDeviceProver(secret, group=self.group)
        public_key = prover_setup.public_key

        self.db.store_user(username, helper_data, commitment, public_key)
        total_time = (time.perf_counter() - t0) * 1000

        return {
            "success": True,
            "username": username,
            "total_time_ms": round(total_time, 2),
            "minutiae_detected": len(minutiae),
            "helper_data_preview": helper_data[:32].tolist(),
            "commitment_hash": commitment,
            "public_key_hex": hex(public_key),
            "storage_path": self.db.db_path
        }

    def authenticate_user(self, username, query_file_param):
        record = self.db.get_user(username)
        if record is None:
            return {"success": False, "error": f"User '{username}' is not enrolled in database."}

        img, err = self.load_image(query_file_param)
        if err:
            return {"success": False, "error": err}

        t_start = time.perf_counter()
        
        # Step 1: Preprocessing & Query Minutiae
        t0_prep = time.perf_counter()
        prep = preprocess_fingerprint(img)
        minutiae = extract_minutiae(prep["skeleton"], orientations=prep["orientations"], mask=prep["mask"])
        query_template = build_template(img, prep, minutiae, vector_bits=256)
        t_prep_ms = (time.perf_counter() - t0_prep) * 1000

        # Step 2: Fuzzy Commitment Recovery
        t0_fc = time.perf_counter()
        helper_data = record["helper_data"]
        stored_commitment = record["commitment"]
        public_key = record["public_key"]

        fc_res = self.fc.recover_secret(query_template.binary_vector, helper_data, stored_commitment)
        t_fc_ms = (time.perf_counter() - t0_fc) * 1000

        if not fc_res["success"]:
            total_time = (time.perf_counter() - t_start) * 1000
            return {
                "authenticated": False,
                "reason": "Fuzzy Commitment decoding failed: bit error rate exceeds ECC error tolerance threshold.",
                "fc_success": False,
                "errors_corrected": fc_res.get("errors_corrected", 0),
                "total_time_ms": round(total_time, 2),
                "minutiae_detected": len(minutiae),
                "timings": {
                    "preprocessing_ms": round(t_prep_ms, 2),
                    "fuzzy_commitment_ms": round(t_fc_ms, 2),
                    "zkp_prover_ms": 0.0,
                    "zkp_verifier_ms": 0.0
                }
            }

        recovered_secret = fc_res["recovered_secret"]

        # Step 3: Schnorr Sigma Zero-Knowledge Proof Protocol
        t0_zkp_p = time.perf_counter()
        prover = UserDeviceProver(recovered_secret, group=self.group)
        t_commit = prover.step1_create_commitment()
        t_zkp_p1 = (time.perf_counter() - t0_zkp_p) * 1000

        t0_zkp_v = time.perf_counter()
        verifier = ServerVerifier(public_key, group=self.group)
        c_challenge = verifier.step1_issue_challenge()
        t_zkp_v1 = (time.perf_counter() - t0_zkp_v) * 1000

        t0_zkp_p2 = time.perf_counter()
        s_response = prover.step2_respond_to_challenge(c_challenge)
        t_zkp_p2 = (time.perf_counter() - t0_zkp_p2) * 1000

        t0_zkp_v2 = time.perf_counter()
        is_valid = verifier.step2_verify(t_commit, s_response, challenge_c=c_challenge)
        t_zkp_v2 = (time.perf_counter() - t0_zkp_v2) * 1000

        total_time = (time.perf_counter() - t_start) * 1000

        return {
            "authenticated": is_valid,
            "reason": "ZKP equation verified g^s == t * y^c (mod p)" if is_valid else "ZKP equation verification failed",
            "fc_success": True,
            "errors_corrected": fc_res["errors_corrected"],
            "zkp_details": {
                "prover_commitment_t": hex(t_commit),
                "verifier_challenge_c": hex(c_challenge),
                "prover_response_s": hex(s_response),
                "public_key_y": hex(public_key),
                "group_prime_bits": self.group.p.bit_length(),
                "order_q_bits": self.group.q.bit_length()
            },
            "total_time_ms": round(total_time, 2),
            "minutiae_detected": len(minutiae),
            "timings": {
                "preprocessing_ms": round(t_prep_ms, 2),
                "fuzzy_commitment_ms": round(t_fc_ms, 2),
                "zkp_prover_ms": round(t_zkp_p1 + t_zkp_p2, 3),
                "zkp_verifier_ms": round(t_zkp_v1 + t_zkp_v2, 3)
            }
        }

    def list_users(self):
        users = self.db.list_users()
        return [{"username": u, "commitment": c, "enrolled_at": ts} for u, c, ts in users]

    def inspect_user(self, username):
        record = self.db.get_user(username)
        if not record:
            return None
        return {
            "username": username,
            "enrolled_at": record["enrolled_at"],
            "helper_data_bits": len(record["helper_data"]),
            "helper_data_preview": record["helper_data"][:64].tolist(),
            "commitment_hash": record["commitment"],
            "public_key_hex": hex(record["public_key"]),
            "db_path": self.db.db_path
        }

backend = BiometricBackend()

class RequestHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            index_file = os.path.join(BASE_DIR, "web", "index.html")
            if os.path.exists(index_file):
                with open(index_file, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        if path == "/api/samples":
            samples = backend.get_samples()
            self._send_json({"samples": samples})
            return

        if path == "/api/users":
            users = backend.list_users()
            self._send_json({"users": users})
            return

        if path == "/api/inspect":
            uname = params.get("username", [None])[0]
            if not uname:
                self._send_json({"error": "Missing username"}, 400)
                return
            rec = backend.inspect_user(uname)
            if not rec:
                self._send_json({"error": f"User {uname} not found"}, 404)
                return
            self._send_json(rec)
            return

        if path == "/api/sample-image":
            file_param = params.get("file", [None])[0]
            if not file_param:
                self.send_error(400, "Missing file parameter")
                return
            img, err = backend.load_image(file_param)
            if err:
                self.send_error(404, err)
                return
            _, buf = cv2.imencode('.png', img)
            png_bytes = buf.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_bytes)))
            self.end_headers()
            self.wfile.write(png_bytes)
            return

        if path == "/api/figures":
            fig_dir = os.path.join(BASE_DIR, "results", "figures")
            figs = []
            if os.path.exists(fig_dir):
                for f in sorted(os.listdir(fig_dir)):
                    if f.endswith(".png"):
                        figs.append(f)
            self._send_json({"figures": figs})
            return

        if path == "/api/figure":
            file_param = params.get("file", [None])[0]
            if not file_param:
                self.send_error(400, "Missing file param")
                return
            safe_name = os.path.basename(file_param)
            fig_path = os.path.join(BASE_DIR, "results", "figures", safe_name)
            if not os.path.exists(fig_path):
                self.send_error(404, "Figure not found")
                return
            with open(fig_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if path in ("/download-report.pdf", "/api/download-report"):
            pdf_path = os.path.join(BASE_DIR, "results", "Secure_Fingerprint_ZKP_Project_Report.pdf")
            if not os.path.exists(pdf_path):
                self.send_error(404, "PDF Report not generated yet")
                return
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Disposition", 'attachment; filename="Secure_Fingerprint_ZKP_Project_Report.pdf"')
            self.send_header("Content-Length", str(len(pdf_bytes)))
            self.end_headers()
            self.wfile.write(pdf_bytes)
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = {}
        if body:
            try:
                data = json.loads(body.decode('utf-8'))
            except Exception:
                pass

        if path == "/api/analyze":
            file_param = data.get("file")
            if not file_param:
                self._send_json({"error": "Missing 'file' in request body"}, 400)
                return
            img, err = backend.load_image(file_param)
            if err:
                self._send_json({"error": err}, 404)
                return
            analysis = backend.analyze_fingerprint(img)
            self._send_json(analysis)
            return

        if path == "/api/enroll":
            uname = data.get("username", "").strip()
            file_param = data.get("file", "").strip()
            if not uname or not file_param:
                self._send_json({"error": "username and file are required"}, 400)
                return
            res = backend.enroll_user(uname, file_param)
            self._send_json(res)
            return

        if path == "/api/authenticate":
            uname = data.get("username", "").strip()
            file_param = data.get("file", "").strip()
            if not uname or not file_param:
                self._send_json({"error": "username and file are required"}, 400)
                return
            res = backend.authenticate_user(uname, file_param)
            self._send_json(res)
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def run(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"[*] Biometric ZKP Web Server running at http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port=port)
