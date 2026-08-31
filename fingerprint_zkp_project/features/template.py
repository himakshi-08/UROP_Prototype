"""
Biometric Template Generator.
Generates:
1. Minutiae-based template (list of minutiae points for baseline matching).
2. Fixed-length error-tolerant binary vector B for Fuzzy Commitment protection.
"""

import numpy as np

class BiometricTemplate:
    def __init__(self, minutiae, image_shape=(300, 300), grid_size=(16, 16), vector_bits=256):
        self.minutiae = minutiae
        self.image_shape = image_shape
        self.grid_size = grid_size
        self.vector_bits = vector_bits
        self.binary_vector = self.to_binary_vector()

    def to_binary_vector(self):
        """
        Converts minutiae set into a stable, fixed-length binary string B of size `vector_bits`.
        Uses spatial grid quantization over image ROI to maximize intra-class stability
        and error-tolerance under minor translation/rotation.
        """
        h, w = self.image_shape
        rows, cols = self.grid_size
        bits_per_cell = self.vector_bits // (rows * cols)
        if bits_per_cell < 1:
            bits_per_cell = 1
            rows = cols = int(np.sqrt(self.vector_bits))

        binary_vector = np.zeros(self.vector_bits, dtype=np.uint8)

        cell_h = h / rows
        cell_w = w / cols

        if not self.minutiae:
            return binary_vector

        # Calculate centroid to make representation translation-invariant
        mean_x = np.mean([m.x for m in self.minutiae])
        mean_y = np.mean([m.y for m in self.minutiae])

        for m in self.minutiae:
            # Shift coordinates relative to center of mass
            aligned_x = (m.x - mean_x) + (w / 2.0)
            aligned_y = (m.y - mean_y) + (h / 2.0)

            r_idx = int(aligned_y / cell_h)
            c_idx = int(aligned_x / cell_w)

            r_idx = min(max(r_idx, 0), rows - 1)
            c_idx = min(max(c_idx, 0), cols - 1)

            base_bit_idx = (r_idx * cols + c_idx) * bits_per_cell
            if base_bit_idx < self.vector_bits:
                binary_vector[base_bit_idx] = 1 # Minutia presence bit

                if bits_per_cell >= 2:
                    # Type bit (1 for bifurcation, 0 for ending)
                    binary_vector[base_bit_idx + 1] = 1 if m.type == "bifurcation" else 0

                if bits_per_cell >= 4:
                    # Quantized angle orientation (2 bits for 4 quadrants)
                    norm_angle = (m.orientation % np.pi) / np.pi
                    angle_bin = int(norm_angle * 4) % 4
                    binary_vector[base_bit_idx + 2] = (angle_bin >> 1) & 1
                    binary_vector[base_bit_idx + 3] = angle_bin & 1

        return binary_vector

    def get_binary_string(self):
        return "".join(str(b) for b in self.binary_vector)

    @staticmethod
    def hamming_distance(vec1, vec2):
        """
        Calculates Hamming distance between two equal-length binary vectors.
        """
        return np.sum(vec1 != vec2)

    @staticmethod
    def hamming_similarity(vec1, vec2):
        """
        Calculates normalized Hamming similarity (1 - normalized distance).
        """
        dist = BiometricTemplate.hamming_distance(vec1, vec2)
        return 1.0 - (dist / float(len(vec1)))
