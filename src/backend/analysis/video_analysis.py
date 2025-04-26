from backend.analysis.base_analyzer import BaseAnalyzer
from backend.analysis.video_utils.face_analyzer import FaceAnalyzerUtil # Import the new utility
import os

class VideoAnalyzer(BaseAnalyzer):
    """Analyzes videos using FaceAnalyzerUtil for face warping and blink inconsistencies."""

    def analyze(self, file_path: str) -> dict:
        """
        Performs video analysis using the FaceAnalyzerUtil.

        Args:
            file_path: Path to the video file.

        Returns:
            A dictionary with analysis results.
        """
        if not os.path.exists(file_path):
             return {"error": f"File not found: {file_path}"}

        face_util = None # Initialize to ensure finally block works
        try:
            # Instantiate the utility class
            face_util = FaceAnalyzerUtil()
            # Process the video using the utility
            analysis_data = face_util.process_video(file_path)

            # Check if the utility returned an error
            if "error" in analysis_data:
                return analysis_data # Propagate the error

            # --- Interpret the results from the utility ---
            duration = analysis_data.get("duration_seconds", 0)
            blink_rate_hz = analysis_data.get("blink_rate_hz", 0)
            landmark_stability = analysis_data.get("landmark_detection_ratio", 0)
            processed_frames = analysis_data.get("processed_frames", 0)

            # Heuristic: Suspicious blink rate
            # Normal human blink rate is ~0.2-0.5 Hz (12-30 blinks/min)
            blink_rate_suspicious = (duration > 1 and (blink_rate_hz < 0.1 or blink_rate_hz > 1.0))

            # Heuristic: Potential warping if landmarks are lost frequently
            # Suspicious if landmarks detected in less than 80% of frames
            potential_face_warping = landmark_stability < 0.80 and processed_frames > 0

            # --- Construct the final result dictionary ---
            result = {
                "analyzer": "Video Analysis (MediaPipe Face Mesh)",
                "file": file_path,
                "frame_count": analysis_data.get("total_frames", 0),
                "processed_frames": processed_frames,
                "duration_seconds": duration,
                "total_blinks_detected": analysis_data.get("total_blinks_detected", 0),
                "blink_rate_hz": blink_rate_hz,
                "suspicious_blink_rate": blink_rate_suspicious, # Renamed from heuristic
                "landmark_detection_ratio": landmark_stability,
                "potential_face_warping": potential_face_warping, # Renamed from heuristic
            }
            return result

        except Exception as e:
            import traceback
            print(f"Video analysis orchestration failed: {e}")
            traceback.print_exc()
            return {"error": f"Video analysis orchestration failed: {e}"}
        finally:
            # Ensure MediaPipe resources are released by closing the utility instance
            if face_util:
                face_util.close()
