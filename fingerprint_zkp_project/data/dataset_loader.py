"""
Dataset Loader & Synthetic Fingerprint Generator for UROP Project.
Handles loading raw fingerprint datasets (e.g. FVC format: {subject_id}_{sample_id}.tif/.png)
and generates synthetic baseline datasets for local development/testing.
"""

import os
import glob
import re
import numpy as np
import cv2

class FingerprintDataset:
    def __init__(self, data_dir=None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data", "raw")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_dataset(self):
        """
        Loads fingerprint images from data_dir.
        Files are expected to follow naming standard: {subject_id}_{sample_id}.ext
        Example: 101_1.tif -> Subject 101, Sample 1
        Returns:
            dict: {subject_id: {sample_id: image_matrix}}
        """
        valid_exts = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_paths = []
        for ext in valid_exts:
            image_paths.extend(glob.glob(os.path.join(self.data_dir, ext)))
            image_paths.extend(glob.glob(os.path.join(self.data_dir, ext.upper())))

        if not image_paths:
            print(f"[DatasetLoader] No fingerprint images found in {self.data_dir}. Generating synthetic dataset...")
            self.generate_synthetic_dataset(num_subjects=10, samples_per_subject=8)
            return self.load_dataset()

        dataset = {}
        pattern = re.compile(r'(\d+)_(\d+)')

        for path in sorted(image_paths):
            filename = os.path.basename(path)
            name_part = os.path.splitext(filename)[0]
            match = pattern.search(name_part)
            if match:
                subject_id = int(match.group(1))
                sample_id = int(match.group(2))
            else:
                # Fallback parser if naming convention is non-standard
                subject_id = hash(name_part[:len(name_part)//2]) % 1000
                sample_id = hash(name_part) % 10

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if subject_id not in dataset:
                    dataset[subject_id] = {}
                dataset[subject_id][sample_id] = img

        print(f"[DatasetLoader] Successfully loaded {sum(len(v) for v in dataset.values())} images across {len(dataset)} subjects.")
        return dataset

    def generate_synthetic_dataset(self, num_subjects=10, samples_per_subject=8, img_shape=(300, 300)):
        """
        Generates realistic synthetic fingerprint patterns using Gabor wavelets and ridge patterns
        with subject-specific core patterns and intra-class noise/rotation/translation.
        """
        print(f"[DatasetLoader] Generating synthetic fingerprint dataset ({num_subjects} subjects, {samples_per_subject} samples/subject)...")
        os.makedirs(self.data_dir, exist_ok=True)
        h, w = img_shape

        for s in range(1, num_subjects + 1):
            # Base parameters for subject s
            base_freq = 0.08 + (s % 5) * 0.015
            center_x, center_y = w // 2 + (s * 7) % 20 - 10, h // 2 + (s * 11) % 20 - 10
            spiral_factor = 0.5 + (s * 0.3) % 1.5

            for sample in range(1, samples_per_subject + 1):
                # Intra-class variations (rotation, translation, minor noise)
                angle_offset = (np.random.normal(0, 0.08)) # radians
                dx = np.random.randint(-5, 6)
                dy = np.random.randint(-5, 6)

                y_grid, x_grid = np.ogrid[:h, :w]
                xc = x_grid - (center_x + dx)
                yc = y_grid - (center_y + dy)

                r = np.sqrt(xc**2 + yc**2) + 1e-5
                theta = np.arctan2(yc, xc) + angle_offset

                # Ridge phase pattern
                phase = 2 * np.pi * base_freq * (r + spiral_factor * theta * 10)
                ridges = np.sin(phase)

                # Convert to 8-bit image with background mask and noise
                img = ((ridges + 1) / 2.0 * 200 + 30).astype(np.uint8)
                
                # Circular fingerprint boundary / elliptical mask
                ellipse_mask = ((xc / (w * 0.4))**2 + (yc / (h * 0.45))**2) <= 1.0
                img[~ellipse_mask] = 255 # White background

                # Add Gaussian noise
                noise = np.random.normal(0, 8, img.shape).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                filename = f"{s:03d}_{sample}.png"
                filepath = os.path.join(self.data_dir, filename)
                cv2.imwrite(filepath, img)

        print(f"[DatasetLoader] Synthetic dataset successfully written to {self.data_dir}")

if __name__ == "__main__":
    loader = FingerprintDataset()
    ds = loader.load_dataset()
    print("Dataset subjects:", list(ds.keys()))
