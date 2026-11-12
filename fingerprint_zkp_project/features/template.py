"""
Biometric Template Generator.
Generates:
1. Minutiae-based template (list of minutiae points for baseline matching).
2. Fixed-length error-tolerant binary vector B for Fuzzy Commitment protection.

Binary encoding (256 bits):
  Jain-style Gabor FingerCode. The fingerprint is registered to the mask
  centroid and dominant ridge direction, mapped to a polar tessellation
  (16 sectors x 2 rings), and filtered with an 8-angle Gabor bank.
  Each cell/filter is 1-bit quantized against that filter's median energy.
  Residual rotation is a cyclic sector shift of B (16 bits per sector).
"""

import cv2
import numpy as np


N_SECTORS = 16
N_RINGS = 8
N_FILTERS = 8
BITS_PER_SECTOR = 16  # 8 rings x 2 bits
GABOR_BANK = None
QUADRATURE_GABOR_BANK = None


def _quadrature_gabor_bank(ksize=21, sigma=4.0, lambd=8.0, gamma=0.5):
    """
    Constructs an 8-orientation quadrature Gabor filter bank (even and odd pairs).
    Orientations theta_k = k * pi / 8 for k in 0..7.
    Returns:
        list of (even_kernel, odd_kernel, theta) tuples.
    """
    global QUADRATURE_GABOR_BANK
    if QUADRATURE_GABOR_BANK is not None:
        return QUADRATURE_GABOR_BANK
    bank = []
    for k in range(N_FILTERS):
        theta = float(k) * np.pi / float(N_FILTERS)
        # Even (cosine, psi=0)
        ke = cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi / 2.0, lambd, gamma, 0.0, ktype=cv2.CV_32F)
        # Odd (sine, psi=pi/2)
        ko = cv2.getGaborKernel((ksize, ksize), sigma, theta + np.pi / 2.0, lambd, gamma, np.pi / 2.0, ktype=cv2.CV_32F)
        # Zero-mean DC removal
        ke -= ke.mean()
        ko -= ko.mean()
        bank.append((ke, ko, theta))
    QUADRATURE_GABOR_BANK = bank
    return bank


def _gabor_bank(ksize=21):
    return [item[0] for item in _quadrature_gabor_bank(ksize=ksize)]


def iter_binary_alignments(vec, n_sectors=N_SECTORS, bits_per_sector=BITS_PER_SECTOR):
    vec = np.asarray(vec, dtype=np.uint8)
    seen = set()
    for s in range(n_sectors):
        shifted = np.roll(vec, s * bits_per_sector)
        key = shifted.tobytes()
        if key in seen:
            continue
        seen.add(key)
        yield shifted


class BiometricTemplate:
    def __init__(
        self,
        minutiae,
        image_shape=(300, 300),
        grid_size=(16, 16),
        vector_bits=256,
        orientations=None,
        mask=None,
        source_image=None,
        enhanced=None,
    ):
        self.minutiae = minutiae
        self.image_shape = image_shape
        self.grid_size = grid_size
        self.vector_bits = vector_bits
        self.orientations = orientations
        self.mask = mask
        self.source_image = source_image
        self.enhanced = enhanced
        self.coverage_vector = np.ones(self.vector_bits, dtype=np.float32)
        self.binary_vector = self.to_binary_vector()

    def to_binary_vector(self):
        img = self.enhanced if self.enhanced is not None else self.source_image
        if img is not None and self.orientations is not None:
            return self._gabor_fingercode(img)
        if self.orientations is not None:
            return self._orientation_field_vector()
        return self._minutiae_occupancy_vector()

    def _prepare_polar(self, img):
        img_f = np.asarray(img, dtype=np.float32)
        h, w = img_f.shape[:2]
        ori = np.asarray(self.orientations, dtype=np.float64)
        if ori.shape != img_f.shape:
            ori = cv2.resize(ori, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.mask is not None:
            mask = np.asarray(self.mask).astype(np.uint8)
            if mask.shape != img_f.shape:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8)

        ys, xs = np.where(mask > 0)
        if xs.size < 40:
            return None

        cx, cy = float(xs.mean()), float(ys.mean())
        ang = ori[mask > 0]
        mean_th = 0.5 * np.arctan2(np.mean(np.sin(2.0 * ang)), np.mean(np.cos(2.0 * ang)))
        rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(mean_th), 1.0)
        img_w = cv2.warpAffine(img_f, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask_w = cv2.warpAffine(mask.astype(np.float32), rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)

        # Step 1: Double-Angle Orientation Field Transformation
        # Represent axial orientations as continuous unit vectors u = cos(2*ori), v = sin(2*ori)
        u = np.cos(2.0 * ori).astype(np.float32)
        v = np.sin(2.0 * ori).astype(np.float32)
        # Warp u and v scalar fields independently with linear interpolation (avoids 0 <-> pi jump)
        u_w = cv2.warpAffine(u, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        v_w = cv2.warpAffine(v, rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        # Apply the -mean_th rotation to the double-angle vectors (-2*mean_th in 2*theta space)
        cos_2a = float(np.cos(2.0 * mean_th))
        sin_2a = float(np.sin(2.0 * mean_th))
        u_rot = u_w * cos_2a + v_w * sin_2a
        v_rot = -u_w * sin_2a + v_w * cos_2a
        # Reconstruct transformed orientation field modulo pi
        ori_w = np.mod(0.5 * np.arctan2(v_rot, u_rot), np.pi).astype(np.float32)

        ys2, xs2 = np.where(mask_w > 0.35)
        if xs2.size < 40:
            return None
        dx = xs2.astype(np.float64) - cx
        dy = ys2.astype(np.float64) - cy
        radii = np.sqrt(dx * dx + dy * dy)
        max_r = max(float(np.percentile(radii, 88)), 28.0)
        inner = 0.18 * max_r
        return img_w, mask_w, ori_w, (cx, cy), max_r, inner

    def _gabor_fingercode(self, img):
        bits = np.zeros(self.vector_bits, dtype=np.uint8)
        prepared = self._prepare_polar(img)
        if prepared is None:
            return bits
        img_w, mask_w, ori_w, center, max_r, inner = prepared
        h, w = img_w.shape
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        ring_mask = ((rr >= inner) & (rr <= max_r) & (mask_w > 0.25)).astype(np.float32)
        img_roi = img_w * ring_mask
        n_sec, n_ring = 16, 8
        dsize = (n_ring, n_sec)
        flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_AREA

        # Step 3: Convolve with the 8-orientation quadrature Gabor bank
        gabor_bank = _quadrature_gabor_bank()
        p_energies = []
        thetas = []
        for ke, ko, theta_k in gabor_bank:
            re = cv2.filter2D(img_roi, cv2.CV_32F, ke)
            ro = cv2.filter2D(img_roi, cv2.CV_32F, ko)
            energy_k = np.sqrt(re * re + ro * ro) * ring_mask
            polar_e_k = cv2.warpPolar(energy_k, dsize, center, max_r, flags)
            p_energies.append(polar_e_k)
            thetas.append(theta_k)

        polar_m = cv2.warpPolar(ring_mask, dsize, center, max_r, flags)
        valid_cells = (polar_m > 0.2)

        # 1. Total Directional Ridge Energy per polar cell
        E_total = np.sum(p_energies, axis=0)
        med_energy = float(np.median(E_total[valid_cells])) if np.any(valid_cells) else float(np.median(E_total))
        bits_i = (E_total > med_energy).astype(np.uint8)

        # 2. Dominant Gabor Orientation via vector sum across the 8 orientations
        U = np.sum([p_energies[k] * np.cos(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
        V = np.sum([p_energies[k] * np.sin(2.0 * thetas[k]) for k in range(N_FILTERS)], axis=0)
        theta_gabor = np.mod(0.5 * np.arctan2(V, U), np.pi).astype(np.float32)
        bits_o = (theta_gabor > (np.pi / 2.0)).astype(np.uint8)

        out = []
        coverage = []
        for s in range(n_sec):
            for r in range(n_ring):
                cov = float(polar_m[s, r])
                out.append(int(bits_i[s, r]))
                coverage.append(cov)
                out.append(int(bits_o[s, r]))
                coverage.append(cov)

        bits[: min(self.vector_bits, len(out))] = np.array(out[: self.vector_bits], dtype=np.uint8)
        self.coverage_vector = np.array(coverage[: self.vector_bits], dtype=np.float32)
        return bits

    def _orientation_field_vector(self):
        prepared = self._prepare_polar(np.asarray(self.orientations, dtype=np.float32))
        bits = np.zeros(self.vector_bits, dtype=np.uint8)
        if prepared is None:
            return bits
        ori_w, mask_w, _ori2, center, max_r, _inner = prepared
        cos2 = np.cos(2.0 * ori_w)
        sin2 = np.sin(2.0 * ori_w)
        dsize = (N_RINGS * 4, N_SECTORS * 2)
        flags = cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR
        cos_p = cv2.warpPolar(cos2, dsize, center, max_r, flags)
        sin_p = cv2.warpPolar(sin2, dsize, center, max_r, flags)
        mu = 0.5 * np.arctan2(sin_p, cos_p)
        mu = np.mod(mu, np.pi)
        # Downsample to 16x8 2-bit = 256
        mu_s = cv2.resize(mu, (16, 8), interpolation=cv2.INTER_AREA)
        out = []
        for r in range(8):
            for c in range(16):
                q = int((mu_s[r, c] / np.pi) * 4.0) % 4
                g = q ^ (q >> 1)
                out.extend([(g >> 0) & 1, (g >> 1) & 1])
        bits[: min(self.vector_bits, len(out))] = np.array(out[: self.vector_bits], dtype=np.uint8)
        return bits

    def _minutiae_occupancy_vector(self):
        rows, cols = self.grid_size
        bits = np.zeros(self.vector_bits, dtype=np.uint8)
        if not self.minutiae:
            return bits

        xs = np.array([m.x for m in self.minutiae], dtype=np.float64)
        ys = np.array([m.y for m in self.minutiae], dtype=np.float64)
        thetas = np.array([m.orientation for m in self.minutiae], dtype=np.float64)
        cx, cy = float(xs.mean()), float(ys.mean())
        mean_th = 0.5 * np.arctan2(np.mean(np.sin(2.0 * thetas)), np.mean(np.cos(2.0 * thetas)))
        dx, dy = xs - cx, ys - cy
        c, s = np.cos(-mean_th), np.sin(-mean_th)
        rx, ry = c * dx - s * dy, s * dx + c * dy
        std_x = max(float(np.std(rx)), 8.0)
        std_y = max(float(np.std(ry)), 8.0)
        nx = np.clip((rx / (3.0 * std_x) + 1.0) * 0.5, 0.0, 0.9999)
        ny = np.clip((ry / (3.0 * std_y) + 1.0) * 0.5, 0.0, 0.9999)
        occ = np.zeros((rows, cols), dtype=np.uint8)
        for r, col in zip((ny * rows).astype(np.int32), (nx * cols).astype(np.int32)):
            occ[r, col] = 1
        flat = occ.reshape(-1)
        bits[: min(self.vector_bits, flat.size)] = flat[: self.vector_bits]
        return bits

    def get_validity_mask(self, tau=0.7):
        return (self.coverage_vector >= tau).astype(np.uint8)

    @staticmethod
    def masked_hamming_distance(v1, v2, m1, m2, min_overlap=0.50):
        v1 = np.asarray(v1, dtype=np.uint8)
        v2 = np.asarray(v2, dtype=np.uint8)
        m = (np.asarray(m1) > 0) & (np.asarray(m2) > 0)
        n_valid = int(np.sum(m))
        total_bits = len(v1)
        overlap = n_valid / float(total_bits) if total_bits > 0 else 0.0
        if overlap < min_overlap or n_valid == 0:
            return None
        errs = int(np.sum(v1[m] != v2[m]))
        norm_hd = errs / float(n_valid)
        equiv_bits = norm_hd * float(total_bits)
        return {
            "norm_hd": float(norm_hd),
            "equiv_bits": float(equiv_bits),
            "errs": errs,
            "n_valid": n_valid,
            "overlap": float(overlap)
        }

    @staticmethod
    def aligned_masked_hamming_distance(v1, v2, m1, m2, n_sectors=N_SECTORS, bits_per_sector=BITS_PER_SECTOR, min_overlap=0.50):
        v1 = np.asarray(v1, dtype=np.uint8)
        v2 = np.asarray(v2, dtype=np.uint8)
        m1 = np.asarray(m1, dtype=np.uint8)
        m2 = np.asarray(m2, dtype=np.uint8)

        best_res = None
        for s in range(n_sectors):
            sh_v2 = np.roll(v2, s * bits_per_sector)
            sh_m2 = np.roll(m2, s * bits_per_sector)
            res = BiometricTemplate.masked_hamming_distance(v1, sh_v2, m1, sh_m2, min_overlap=min_overlap)
            if res is not None:
                if best_res is None or res["norm_hd"] < best_res["norm_hd"]:
                    best_res = dict(res)
                    best_res["shift_sector"] = s

        return best_res

    def get_binary_string(self):
        return "".join(str(int(b)) for b in self.binary_vector)

    @staticmethod
    def hamming_distance(vec1, vec2):
        v1 = np.array(vec1, dtype=np.uint8)
        v2 = np.array(vec2, dtype=np.uint8)
        n = min(len(v1), len(v2))
        return int(np.sum(v1[:n] != v2[:n]))

    @staticmethod
    def aligned_hamming_distance(vec1, vec2):
        v1 = np.array(vec1, dtype=np.uint8)
        best = BiometricTemplate.hamming_distance(v1, vec2)
        for shifted in iter_binary_alignments(vec2):
            hd = int(np.sum(v1 != shifted))
            if hd < best:
                best = hd
        return int(best)

    @staticmethod
    def hamming_similarity(vec1, vec2):
        dist = BiometricTemplate.aligned_hamming_distance(vec1, vec2)
        n = min(len(vec1), len(vec2))
        if n == 0:
            return 0.0
        return 1.0 - (dist / float(n))


def build_template(img, prep, minutiae, vector_bits=256):
    return BiometricTemplate(
        minutiae,
        image_shape=img.shape,
        vector_bits=vector_bits,
        orientations=prep.get("orientations"),
        mask=prep.get("mask"),
        source_image=img,
        enhanced=prep.get("enhanced"),
    )
