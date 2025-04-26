from backend.analysis.base_analyzer import BaseAnalyzer
import cv2
import numpy as np
import os

class ChromaticAberrationAnalyzer(BaseAnalyzer):
    """
    Analyzes potential chromatic aberration by measuring differences
    in edge gradients across color channels.
    Note: This is an approximation and its effectiveness for deepfake
    detection may be limited.
    """

    def analyze(self, file_path: str) -> dict:
        """
        Performs analysis based on color channel gradient differences at edges.

        Args:
            file_path: Path to the image file.

        Returns:
            A dictionary with analysis results.
        """
        if not os.path.exists(file_path):
             return {"error": f"File not found: {file_path}"}
        try:
            img = cv2.imread(file_path)
            if img is None:
                return {"error": f"Could not read image: {file_path}"}

            # Ensure image is 3-channel color
            if len(img.shape) != 3 or img.shape[2] != 3:
                return {"error": "Image is not a 3-channel color image."}

            # --- More Sophisticated Analysis ---
            # 1. Calculate gradients for each color channel
            # Use Scharr operator for potentially higher accuracy than Sobel 3x3
            grad_x_b = cv2.Scharr(img[:,:,0], cv2.CV_64F, 1, 0)
            grad_y_b = cv2.Scharr(img[:,:,0], cv2.CV_64F, 0, 1)
            grad_x_g = cv2.Scharr(img[:,:,1], cv2.CV_64F, 1, 0)
            grad_y_g = cv2.Scharr(img[:,:,1], cv2.CV_64F, 0, 1)
            grad_x_r = cv2.Scharr(img[:,:,2], cv2.CV_64F, 1, 0)
            grad_y_r = cv2.Scharr(img[:,:,2], cv2.CV_64F, 0, 1)

            # 2. Calculate gradient magnitude for each channel
            mag_b = cv2.magnitude(grad_x_b, grad_y_b)
            mag_g = cv2.magnitude(grad_x_g, grad_y_g)
            mag_r = cv2.magnitude(grad_x_r, grad_y_r)

            # 3. Identify strong edge regions (using Green channel magnitude as reference)
            # Normalize green magnitude for thresholding
            mag_g_norm = cv2.normalize(mag_g, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # Use Otsu's threshold to automatically find a threshold for strong edges
            _, edge_mask = cv2.threshold(mag_g_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 4. Calculate average magnitude difference in edge regions
            num_edge_pixels = np.count_nonzero(edge_mask)

            if num_edge_pixels > 0:
                # Calculate mean absolute difference between channel magnitudes at edges
                diff_rg_mag = np.mean(np.abs(mag_r[edge_mask > 0] - mag_g[edge_mask > 0]))
                diff_gb_mag = np.mean(np.abs(mag_g[edge_mask > 0] - mag_b[edge_mask > 0]))
                diff_rb_mag = np.mean(np.abs(mag_r[edge_mask > 0] - mag_b[edge_mask > 0]))
                avg_mag_diff = (diff_rg_mag + diff_gb_mag + diff_rb_mag) / 3
            else:
                diff_rg_mag = 0
                diff_gb_mag = 0
                diff_rb_mag = 0
                avg_mag_diff = 0

            # --- Results ---
            # Define a heuristic threshold - this is highly empirical and may need tuning
            # A higher difference *might* indicate inconsistencies, but could also be
            # strong natural CA or other image features.
            # Let's use a relative threshold based on average green magnitude at edges
            avg_g_mag_at_edges = np.mean(mag_g[edge_mask > 0]) if num_edge_pixels > 0 else 0
            # Potential issue if average difference is > 10% of average green magnitude? (Very arbitrary)
            threshold_factor = 0.10
            potential_issue = (avg_mag_diff > (avg_g_mag_at_edges * threshold_factor)) if avg_g_mag_at_edges > 0 else False

            return {
                "analyzer": "Chromatic Aberration (Gradient Diff)",
                "file": file_path,
                "mean_abs_diff_rg_magnitude_at_edges": round(diff_rg_mag, 4),
                "mean_abs_diff_gb_magnitude_at_edges": round(diff_gb_mag, 4),
                "mean_abs_diff_rb_magnitude_at_edges": round(diff_rb_mag, 4),
                "average_magnitude_difference_at_edges": round(avg_mag_diff, 4),
                "potential_issue_heuristic": potential_issue,
                "details": f"Analysis based on {num_edge_pixels} edge pixels identified via Otsu threshold on Green channel Scharr magnitude."
            }
        except Exception as e:
            # Log the exception for debugging
            # print(f"Error during Chromatic Aberration analysis for {file_path}: {e}")
            return {"error": f"Chromatic Aberration analysis failed: {str(e)}"}

