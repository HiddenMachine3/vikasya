from backend.analysis.base_analyzer import BaseAnalyzer
from PIL import Image, ImageChops, ImageEnhance
import os
import io

class ELAAnalyzer(BaseAnalyzer):
    """Performs Error Level Analysis (ELA) on JPEG images."""

    def analyze(self, file_path: str, quality=90, scale=15) -> dict:
        """
        Performs ELA on a JPEG image.

        Args:
            file_path: Path to the image file.
            quality: The quality level for re-saving the image during ELA.
            scale: Factor to enhance the ELA result visibility.


        Returns:
            A dictionary with analysis results (placeholder: returns basic info).
            A real implementation would save/return the ELA image itself
            or calculate statistics on it.
        """
        if not os.path.exists(file_path):
             return {"error": f"File not found: {file_path}"}

        try:
            original_image = Image.open(file_path).convert('RGB')

            # Save temporary JPEG version
            temp_buffer = io.BytesIO()
            original_image.save(temp_buffer, 'JPEG', quality=quality)
            temp_buffer.seek(0)
            resaved_image = Image.open(temp_buffer)

            # Calculate ELA difference
            ela_image = ImageChops.difference(original_image, resaved_image)

            # Enhance the ELA image for visibility
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1 # Avoid division by zero
            scale_factor = 255.0 / max_diff * scale
            ela_enhanced = ImageEnhance.Brightness(ela_image).enhance(scale_factor)

            # Placeholder: Return basic info. A real app might save ela_enhanced
            # or calculate metrics on it (e.g., histogram, std dev).
            ela_mean = sum(ela_enhanced.getdata(0)) / len(ela_enhanced.getdata(0)) # Example metric

            return {
                "analyzer": "Error Level Analysis (ELA)",
                "file": file_path,
                "ela_mean_brightness": round(ela_mean, 2),
                # Simple heuristic: higher mean brightness might indicate tampering
                "potential_tampering": ela_mean > 20.0 # Arbitrary threshold
            }
        except Exception as e:
             # Catch potential errors like non-JPEG files or Pillow issues
            return {"error": f"ELA analysis failed: {e}"}
