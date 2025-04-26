from backend.analysis.base_analyzer import BaseAnalyzer
import cv2
import numpy as np
import os

class FractalDensityAnalyzer(BaseAnalyzer):
    """Analyzes fractal density patterns in images using the box-counting method."""

    def _boxcount(self, img_binary, k):
        """Helper function to count boxes covering non-zero pixels."""
        S = np.add.reduceat(
            np.add.reduceat(img_binary, np.arange(0, img_binary.shape[0], k), axis=0),
                                        np.arange(0, img_binary.shape[1], k), axis=1)
        # Count boxes where sum is > 0 (contains part of the fractal)
        return len(np.where(S > 0)[0])

    def analyze(self, file_path: str) -> dict:
        """
        Performs fractal density analysis using the box-counting method on image edges.

        Args:
            file_path: Path to the image file.

        Returns:
            A dictionary with analysis results, including the estimated fractal dimension.
        """
        if not os.path.exists(file_path):
             return {"error": f"File not found: {file_path}"}
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": f"Could not read image: {file_path}"}

            # --- Box-Counting Implementation ---
            # 1. Preprocessing: Use Canny edge detection to get a binary image
            #    Adjust thresholds as needed, these are common starting points.
            edges = cv2.Canny(img, 100, 200)
            if not np.any(edges):
                 # Handle cases with no detectable edges
                 return {
                    "analyzer": "Fractal Density (Box Counting)",
                    "file": file_path,
                    "estimated_fractal_dimension": 0.0, # Or indicate insufficient detail
                    "notes": "No significant edges detected for analysis."
                 }

            # Convert edges to boolean or 0/1 for counting
            img_binary = edges > 0

            # 2. Box Sizes: Use powers of 2, up to half the smallest dimension
            min_dim = min(img_binary.shape)
            n = int(np.floor(np.log2(min_dim)))
            sizes = 2**np.arange(n, 1, -1) # Larger to smaller boxes (e.g., 64, 32, 16, 8, 4)

            # 3. Counting: Count boxes for each size
            counts = []
            for size in sizes:
                counts.append(self._boxcount(img_binary, size))

            if not counts or all(c == 0 for c in counts):
                 return {
                    "analyzer": "Fractal Density (Box Counting)",
                    "file": file_path,
                    "estimated_fractal_dimension": 0.0,
                    "notes": "Could not derive counts for box sizes."
                 }

            # Filter out zero counts to avoid log(0) issues
            valid_indices = [i for i, count in enumerate(counts) if count > 0]
            if len(valid_indices) < 2: # Need at least two points for regression
                 return {
                    "analyzer": "Fractal Density (Box Counting)",
                    "file": file_path,
                    "estimated_fractal_dimension": 0.0,
                    "notes": "Insufficient data points for fractal dimension calculation."
                 }

            filtered_counts = np.array([counts[i] for i in valid_indices])
            filtered_sizes = np.array([sizes[i] for i in valid_indices])

            # 4. Linear Regression: Fit log(count) vs log(1/size)
            #    The slope is the fractal dimension (Hausdorff-Besicovitch dimension estimate)
            coeffs = np.polyfit(np.log(1.0/filtered_sizes), np.log(filtered_counts), 1)
            fractal_dimension_estimate = coeffs[0]

            # --- Result ---
            # Interpretation of fractal dimension for deepfakes is complex and empirical.
            # Natural images often have dimensions between 1.2 and 1.8 for edges.
            # Significant deviation *might* be suspicious, but context is key.
            # The 'authentic' threshold here remains arbitrary.
            is_potentially_authentic = 1.2 < fractal_dimension_estimate < 1.9

            return {
                "analyzer": "Fractal Density (Box Counting)",
                "file": file_path,
                "estimated_fractal_dimension": round(fractal_dimension_estimate, 4),
                "potentially_authentic_range": is_potentially_authentic # Example interpretation
            }
        except Exception as e:
            import traceback
            print(f"Fractal Density analysis failed: {e}")
            traceback.print_exc()
            return {"error": f"Fractal Density analysis failed: {e}"}

