import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import os
import traceback

class FaceAnalyzerUtil:
    """Provides utility functions for face analysis using MediaPipe Face Mesh."""

    def __init__(self, ear_threshold=0.20, ear_consec_frames=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

        self.EAR_THRESHOLD = ear_threshold
        self.EAR_CONSEC_FRAMES = ear_consec_frames

        # Eye landmark indices for EAR calculation (6 points per eye)
        self.LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]

    def _calculate_ear(self, eye):
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C == 0:
            return 0.3 # Avoid division by zero
        ear = (A + B) / (2.0 * C)
        return ear

    def process_video(self, file_path: str) -> dict:
        """
        Processes a video file to detect blinks and landmark stability.

        Args:
            file_path: Path to the video file.

        Returns:
            A dictionary containing analysis results or an error.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.close() # Ensure mediapipe resources are potentially released
            return {"error": f"Could not open video file: {file_path}"}

        frame_count = 0
        total_blinks = 0
        ear_counter = 0
        landmarks_detected_count = 0
        processed_frame_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame_count += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.face_mesh.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if results.multi_face_landmarks:
                    landmarks_detected_count += 1
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = np.array([(lm.x, lm.y) for lm in face_landmarks.landmark])

                    left_eye_pts = landmarks[self.LEFT_EYE_EAR_INDICES]
                    right_eye_pts = landmarks[self.RIGHT_EYE_EAR_INDICES]

                    left_ear = self._calculate_ear(left_eye_pts)
                    right_ear = self._calculate_ear(right_eye_pts)
                    ear = (left_ear + right_ear) / 2.0

                    if ear < self.EAR_THRESHOLD:
                        ear_counter += 1
                    else:
                        if ear_counter >= self.EAR_CONSEC_FRAMES:
                            total_blinks += 1
                        ear_counter = 0
                else:
                    # If no landmarks detected, check if a blink was in progress
                    if ear_counter >= self.EAR_CONSEC_FRAMES:
                         total_blinks += 1
                    ear_counter = 0 # Reset counter if face is lost

                processed_frame_count += 1

            # Final check if video ended during a potential blink
            if ear_counter >= self.EAR_CONSEC_FRAMES:
                total_blinks += 1

            blink_rate_hz = (total_blinks / duration) if duration > 0 else 0
            landmark_stability = (landmarks_detected_count / processed_frame_count) if processed_frame_count > 0 else 0

            return {
                "total_frames": total_frames,
                "processed_frames": processed_frame_count,
                "duration_seconds": round(duration, 2),
                "total_blinks_detected": total_blinks,
                "blink_rate_hz": round(blink_rate_hz, 2),
                "landmark_detection_ratio": round(landmark_stability, 2),
            }

        except Exception as e:
            print(f"Face analysis utility failed during processing: {e}")
            traceback.print_exc()
            return {"error": f"Face analysis utility failed: {e}"}
        finally:
            if cap.isOpened(): cap.release()
            # Note: self.close() should be called explicitly when done with the utility instance

    def close(self):
        """Releases MediaPipe resources."""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            self.face_mesh.close()
            print("MediaPipe Face Mesh resources released.")
