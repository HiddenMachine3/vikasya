import os

from backend.analysis.audio_analysis import AudioDeepfakeAnalyzer
from backend.analysis.chromatic_aberration import ChromaticAberrationAnalyzer
from backend.analysis.ela import ELAAnalyzer
from backend.analysis.fractal_density import FractalDensityAnalyzer
from backend.analysis.video_analysis import VideoAnalyzer
from backend.analysis.deep_image_analyzer import DeepImageAnalyzer

class MediaAnalyzerService:
    """Orchestrates different media analysis techniques."""

    def __init__(self):
        self.image_analyzers = [
            FractalDensityAnalyzer(),
            ELAAnalyzer(),
            ChromaticAberrationAnalyzer(),
            DeepImageAnalyzer(),
        ]
        self.video_analyzers = [
            VideoAnalyzer(),
            # Can also apply image analyzers frame-by-frame if needed,
            # but VideoAnalyzer is intended for temporal analysis.
        ]
        self.audio_analyzers = [AudioDeepfakeAnalyzer()]

    def analyze_media(self, file_path: str) -> list[dict]:
        """
        Analyzes a media file using appropriate analyzers based on file type.

        Args:
            file_path: Path to the media file.

        Returns:
            A list of dictionaries, each containing results from one analyzer.
            Returns an error dictionary if the file type is unsupported or not found.
        """
        file_path = file_path.strip()
        if not os.path.exists(file_path):
            return [{"error": f"File not found: {file_path}"}]

        _, ext = os.path.splitext(file_path.lower())
        results = []

        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            print(f"Analyzing image: {file_path}")
            for analyzer in self.image_analyzers:
                results.append(analyzer.analyze(file_path))
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            print(f"Analyzing video: {file_path}")
            # Add results from video-specific analyzers
            for analyzer in self.video_analyzers:
                results.append(analyzer.analyze(file_path))
            # Optionally, could add frame-based analysis using image analyzers here
            # e.g., analyze first frame with image analyzers
            # results.append(self.image_analyzers[0].analyze(first_frame_path)) # Needs frame extraction
        elif ext in [".wav", ".mp3", ".flac"]:
            print(f"Analyzing audio: {file_path}")
            for analyzer in self.audio_analyzers:
                results.append(analyzer.analyze(file_path))
        else:
            return [{"error": f"Unsupported file type: {ext}"}]

        return results
