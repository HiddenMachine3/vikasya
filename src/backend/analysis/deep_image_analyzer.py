import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from backend.analysis.base_analyzer import BaseAnalyzer

# Constants (Adjust path as needed)
# Ensure this path points to your actual TFLite model for image analysis

TFLITE_MODEL_PATH = "models/best_model_image_analysis_84_val_acc.tflite" # os.path.join(os.path.dirname(__file__), "../../../", "image_analysis_model.tflite")
IMG_SIZE = 224
# Expected feature length based on notebook: LBP (10) + OG (8) + SSEE (2) = 20
EXPECTED_FEATURE_LENGTH = 20

class DeepImageAnalyzer(BaseAnalyzer):
    """
    Analyzes image files to detect potential deepfakes
    using a TFLite model trained for this purpose.
    """

    def __init__(self):
        """Initialize the analyzer with model path and parameters."""
        self.tflite_model_path = TFLITE_MODEL_PATH
        self.img_size = IMG_SIZE
        self.expected_feature_length = EXPECTED_FEATURE_LENGTH
        # Load interpreter once during initialization for efficiency
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._image_input_index = -1
        self._feature_input_index = -1
        self._load_model()

    def _load_model(self):
        """Loads the TFLite model and prepares the interpreter."""
        if not os.path.exists(self.tflite_model_path):
            print(f"ERROR: TFLite model not found at {self.tflite_model_path}", file=sys.stderr)
            return

        try:
            self._interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
            self._interpreter.allocate_tensors()
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            if len(self._input_details) != 2:
                print(f"ERROR: Expected TFLite model to have 2 inputs, but found {len(self._input_details)}.", file=sys.stderr)
                self._interpreter = None # Invalidate interpreter
                return

            # Identify input indices (heuristic based on shape)
            if len(self._input_details[0]['shape']) == 4 and self._input_details[0]['shape'][1] == self.img_size:
                self._image_input_index = self._input_details[0]['index']
                self._feature_input_index = self._input_details[1]['index']
            elif len(self._input_details[1]['shape']) == 4 and self._input_details[1]['shape'][1] == self.img_size:
                self._image_input_index = self._input_details[1]['index']
                self._feature_input_index = self._input_details[0]['index']
            else:
                print("ERROR: Could not reliably determine image/feature input indices based on shape.", file=sys.stderr)
                print("Input details:", self._input_details, file=sys.stderr)
                self._interpreter = None # Invalidate interpreter

        except ValueError as e:
             print(f"ERROR: Failed to load TFLite model or allocate tensors: {str(e)}", file=sys.stderr)
             self._interpreter = None
        except Exception as e:
            print(f"ERROR: An unexpected error occurred loading the TFLite model: {e}", file=sys.stderr)
            self._interpreter = None


    # ------------- Feature extraction functions (adapted from tf_lite_inference_image.py) ----------------
    def _extract_lbp_features(self, gray_image, radius=1, n_points=8):
        try:
            if gray_image.dtype != np.uint8:
                if gray_image.max() <= 1.0 and gray_image.min() >= 0.0:
                    gray_image = (gray_image * 255).astype(np.uint8)
                else:
                    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

            lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
            expected_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=expected_bins, range=(0, expected_bins), density=True)
            if len(hist) == 0: return np.zeros(expected_bins, dtype=np.float32)
            if len(hist) < expected_bins: hist = np.pad(hist, (0, expected_bins - len(hist)), 'constant')
            return hist.astype(np.float32)
        except Exception as e:
            print(f"ERROR: Error processing LBP: {e}", file=sys.stderr)
            return np.zeros(n_points + 2, dtype=np.float32)

    def _extract_og_features(self, gray_image, n_bins=8):
        try:
            if gray_image.dtype != np.uint8:
                if gray_image.max() <= 1.0 and gray_image.min() >= 0.0:
                    gray_image = (gray_image * 255).astype(np.uint8)
                else:
                    gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

            gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            hist, _ = np.histogram(angle.ravel(), bins=n_bins, range=(0, 360), weights=magnitude.ravel(), density=False)
            hist_sum = np.sum(hist)
            if hist_sum > 0: hist = hist / hist_sum
            else: hist = np.zeros(n_bins)
            return hist.astype(np.float32)
        except Exception as e:
            print(f"ERROR: Error in OG extraction: {e}", file=sys.stderr)
            return np.zeros(n_bins, dtype=np.float32)

    def _extract_ssee_features(self, image, gray_image, quality=75):
        try:
            if gray_image.dtype != np.uint8:
                if gray_image.max() <= 1.0 and gray_image.min() >= 0.0:
                    gray_image_u8 = (gray_image * 255).astype(np.uint8)
                else:
                    gray_image_u8 = np.clip(gray_image, 0, 255).astype(np.uint8)
            else:
                gray_image_u8 = gray_image

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', image, encode_param)
            if not result: return np.array([0.0, 0.0], dtype=np.float32)

            decimg_gray = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
            if decimg_gray is None: return np.array([0.0, 0.0], dtype=np.float32)

            if decimg_gray.shape != gray_image_u8.shape:
                decimg_gray = cv2.resize(decimg_gray, (gray_image_u8.shape[1], gray_image_u8.shape[0]))

            diff = cv2.absdiff(gray_image_u8, decimg_gray)
            return np.array([np.mean(diff), np.std(diff)], dtype=np.float32)
        except Exception as e:
            print(f"ERROR: Error in SSEE (JPEG Error) extraction: {e}", file=sys.stderr)
            return np.array([0.0, 0.0], dtype=np.float32)

    # ------------- Preprocessing for TFLite (adapted from tf_lite_inference_image.py) ----------------
    def _preprocess_image_for_tflite(self, image_path):
        """Reads, preprocesses image, and extracts features for the TFLite model."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}. Skipping.", file=sys.stderr)
                return None, None

            # --- Feature Extraction ---
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray_image.dtype != np.uint8:
                gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

            lbp_hist = self._extract_lbp_features(gray_image)
            og_hist = self._extract_og_features(gray_image)
            ssee_feats = self._extract_ssee_features(image, gray_image)

            if lbp_hist is None or og_hist is None or ssee_feats is None:
                print(f"Warning: Feature extraction failed for {image_path}. Skipping.", file=sys.stderr)
                return None, None

            combined_features = np.concatenate((lbp_hist, og_hist, ssee_feats)).astype(np.float32)

            if len(combined_features) != self.expected_feature_length:
                print(f"Warning: Feature length mismatch for {image_path}. Expected {self.expected_feature_length}, got {len(combined_features)}. Padding/truncating.", file=sys.stderr)
                if len(combined_features) > self.expected_feature_length:
                    combined_features = combined_features[:self.expected_feature_length]
                else:
                    combined_features = np.pad(combined_features, (0, self.expected_feature_length - len(combined_features)), 'constant')

            # --- Image Preprocessing ---
            img_resized = cv2.resize(image, (self.img_size, self.img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_preprocessed = resnet_preprocess(img_rgb) # Handles scaling etc.

            return img_preprocessed.astype(np.float32), combined_features

        except Exception as e:
            print(f"ERROR processing {image_path}: {e}", file=sys.stderr)
            return None, None

    def analyze(self, file_path: str) -> dict:
        """
        Analyzes an image file to determine if it's likely a deepfake.

        Args:
            file_path: Path to the image file.

        Returns:
            A dictionary with analysis results.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        if self._interpreter is None:
             return {"error": "TFLite model interpreter is not loaded or invalid."}

        try:
            # Preprocess the image
            img_data, feature_data = self._preprocess_image_for_tflite(file_path)

            if img_data is None or feature_data is None:
                return {"error": f"Preprocessing failed for {file_path}"}

            # Add batch dimension and ensure correct types/shapes
            img_batch = np.expand_dims(img_data, axis=0).astype(self._input_details[self._image_input_index]['dtype'])
            feature_batch = np.expand_dims(feature_data, axis=0).astype(self._input_details[self._feature_input_index]['dtype'])

            # Verify shapes before setting tensors
            expected_img_shape = self._input_details[self._image_input_index]['shape']
            expected_feat_shape = self._input_details[self._feature_input_index]['shape']

            if not np.array_equal(img_batch.shape, expected_img_shape):
                 return {"error": f"Image batch shape mismatch. Expected {expected_img_shape}, got {img_batch.shape}."}
            if not np.array_equal(feature_batch.shape, expected_feat_shape):
                 return {"error": f"Feature batch shape mismatch. Expected {expected_feat_shape}, got {feature_batch.shape}."}

            # Run inference
            self._interpreter.set_tensor(self._image_input_index, img_batch)
            self._interpreter.set_tensor(self._feature_input_index, feature_batch)
            self._interpreter.invoke()
            output_data = self._interpreter.get_tensor(self._output_details[0]['index'])

            # Get prediction (Assuming output is [prob_fake, prob_real] or similar)
            # Adjust indices [0] and [1] based on your model's output definition
            # If output is single value (e.g., sigmoid), adjust logic
            pred_label = np.argmax(output_data, axis=1)[0] # 0 for first class, 1 for second
            confidence = float(output_data[0][pred_label])

            # Assuming label 0 = Fake, label 1 = Real (adjust if needed based on training)
            prediction_str = "REAL" if pred_label == 1 else "FAKE"
            # Handle potential single-output models (sigmoid)
            if len(output_data[0]) == 1:
                fake_score = float(output_data[0][0])
                real_score = 1.0 - fake_score
                prediction_str = "FAKE" if fake_score > 0.5 else "REAL" # Assuming threshold 0.5
                confidence = fake_score if prediction_str == "FAKE" else real_score
            elif len(output_data[0]) == 2:
                 fake_score = float(output_data[0][0]) # Assuming index 0 is fake
                 real_score = float(output_data[0][1]) # Assuming index 1 is real
            else:
                 # Fallback for unexpected output shape
                 fake_score = 0.0
                 real_score = 0.0
                 prediction_str = "UNKNOWN"
                 confidence = 0.0
                 print(f"Warning: Unexpected output shape from model: {output_data.shape}", file=sys.stderr)


            result = {
                "analyzer": "Image Deepfake Detector",
                "file": file_path,
                "prediction": prediction_str,
                "confidence": round(confidence, 4),
                "raw_scores": {
                    "real_score": round(real_score, 4),
                    "fake_score": round(fake_score, 4),
                },
            }
            return result

        except Exception as e:
            # Log the full traceback for debugging
            import traceback
            print(f"ERROR: Image deepfake analysis failed for {file_path}: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
            return {"error": f"Image deepfake analysis failed: {str(e)}"}

