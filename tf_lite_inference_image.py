import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppress TensorFlow info/warning messages
import shutil
import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import sys

# Constants
IMAGE_DATASET_DIR = os.path.join(os.getcwd(), "data/Dataset") # Adjust if your dataset is elsewhere
REAL_DIR = os.path.join(IMAGE_DATASET_DIR, "Real")
FAKE_DIR = os.path.join(IMAGE_DATASET_DIR, "Fake")
TFLITE_MODEL_PATH = "image_analysis_model.tflite"  # Path to your image TFLite model
TEST_FILES_LOCATION = os.path.join(os.getcwd(), "image_test_files")
IMG_SIZE = 224
# Expected feature length based on notebook: LBP (10) + OG (8) + SSEE (2) = 20
EXPECTED_FEATURE_LENGTH = 20

# ------------- Feature extraction functions (from notebook) ----------------

def extract_lbp_features(gray_image, radius=1, n_points=8):
    try:
        # Ensure gray_image is 8-bit unsigned integer type
        if gray_image.dtype != np.uint8:
             # Normalize to 0-255 range if it's not, e.g., if it's float
            if gray_image.max() <= 1.0 and gray_image.min() >= 0.0:
                 gray_image = (gray_image * 255).astype(np.uint8)
            else:
                 # Attempt conversion, might need adjustment based on actual range
                 gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        # Ensure bins cover the full range expected by 'uniform' method (n_points + 2)
        expected_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=expected_bins, range=(0, expected_bins), density=True)
        if len(hist) == 0:
            return np.zeros(expected_bins, dtype=np.float32)
        # Pad if histogram is smaller than expected (shouldn't happen with correct range)
        if len(hist) < expected_bins:
             hist = np.pad(hist, (0, expected_bins - len(hist)), 'constant')
        return hist.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Error processing LBP: {e}", file=sys.stderr)
        # Return zero array with the expected size
        return np.zeros(n_points + 2, dtype=np.float32)


def extract_og_features(gray_image, n_bins=8):
    try:
        # Ensure gray_image is 8-bit unsigned integer type
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
        if hist_sum > 0:
            hist = hist / hist_sum
        else:
            hist = np.zeros(n_bins)
        return hist.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Error in OG extraction: {e}", file=sys.stderr)
        return np.zeros(n_bins, dtype=np.float32)

def extract_ssee_features(image, gray_image, quality=75):
    try:
        # Ensure gray_image is 8-bit unsigned integer type
        if gray_image.dtype != np.uint8:
             if gray_image.max() <= 1.0 and gray_image.min() >= 0.0:
                 gray_image_u8 = (gray_image * 255).astype(np.uint8)
             else:
                 gray_image_u8 = np.clip(gray_image, 0, 255).astype(np.uint8)
        else:
            gray_image_u8 = gray_image

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param)
        if not result:
            print("ERROR: Failed to encode image to JPEG for SSEE.", file=sys.stderr)
            return np.array([0.0, 0.0], dtype=np.float32)

        decimg_gray = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        if decimg_gray is None:
            print("ERROR: Failed to decode JPEG image for SSEE.", file=sys.stderr)
            return np.array([0.0, 0.0], dtype=np.float32)

        if decimg_gray.shape != gray_image_u8.shape:
            decimg_gray = cv2.resize(decimg_gray, (gray_image_u8.shape[1], gray_image_u8.shape[0]))

        diff = cv2.absdiff(gray_image_u8, decimg_gray)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        return np.array([mean_diff, std_diff], dtype=np.float32)
    except Exception as e:
        print(f"ERROR: Error in SSEE (JPEG Error) extraction: {e}", file=sys.stderr)
        return np.array([0.0, 0.0], dtype=np.float32)

# ------------- Data Loading ----------------

def list_image_files(directory):
    """Lists all image files in a directory."""
    files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            files.append(os.path.join(directory, filename))
    return files

def load_data():
    """Loads image file paths, splits them, copies test files, and returns test set."""
    if not os.path.exists(REAL_DIR):
        print(f"ERROR: Real directory not found: {REAL_DIR}")
        return None, None
    if not os.path.exists(FAKE_DIR):
        print(f"ERROR: Fake directory not found: {FAKE_DIR}")
        return None, None

    real_files = list_image_files(REAL_DIR)
    fake_files = list_image_files(FAKE_DIR)

    if not real_files and not fake_files:
        print(f"ERROR: No image files found in {REAL_DIR} or {FAKE_DIR}")
        return None, None

    files = real_files + fake_files
    labels = [1] * len(real_files) + [0] * len(fake_files) # Assuming Real=1, Fake=0 as often in datasets

    if len(files) < 2:
        print("ERROR: Not enough image files to perform a split.")
        return None, None

    # Split data (e.g., 70% train, 15% val, 15% test)
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            files, labels, test_size=0.3, random_state=42, stratify=labels, shuffle=True
        )
        # Split temp into val and test (50% of temp each -> 15% of total)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels, shuffle=True
        )
    except ValueError as e:
         print(f"ERROR during train/test split: {e}. Check if you have samples from both classes.")
         # Fallback: Use all files as test files if split fails
         test_files = files
         test_labels = labels


    # Create test file directory structure
    if os.path.exists(TEST_FILES_LOCATION):
        shutil.rmtree(TEST_FILES_LOCATION) # Clean up old test files
    os.makedirs(TEST_FILES_LOCATION, exist_ok=True)
    real_test_dir = os.path.join(TEST_FILES_LOCATION, "real")
    fake_test_dir = os.path.join(TEST_FILES_LOCATION, "fake")
    os.makedirs(real_test_dir, exist_ok=True)
    os.makedirs(fake_test_dir, exist_ok=True)

    # Copy test files
    print(f"Copying {len(test_files)} test files to {TEST_FILES_LOCATION}...")
    for file_path, label in zip(test_files, test_labels):
        try:
            target_dir = real_test_dir if label == 1 else fake_test_dir
            shutil.copy(file_path, os.path.join(target_dir, os.path.basename(file_path)))
        except Exception as e:
            print(f"Warning: Could not copy file {file_path}: {e}", file=sys.stderr)

    print("Test file copying complete.")
    return test_files, test_labels

# ------------- Preprocessing for TFLite ----------------

def preprocess_image_for_tflite(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """Reads, preprocesses image, and extracts features for the TFLite model."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.", file=sys.stderr)
            return None, None

        # --- Feature Extraction ---
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ensure grayscale is uint8 for feature extractors expecting it
        if gray_image.dtype != np.uint8:
             gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)

        lbp_hist = extract_lbp_features(gray_image)
        og_hist = extract_og_features(gray_image)
        ssee_feats = extract_ssee_features(image, gray_image) # Pass original color image too

        # Check lengths before concatenation
        if lbp_hist is None or og_hist is None or ssee_feats is None:
             print(f"Warning: Feature extraction failed for {image_path}. Skipping.", file=sys.stderr)
             return None, None

        combined_features = np.concatenate((lbp_hist, og_hist, ssee_feats)).astype(np.float32)

        # Verify feature length
        if len(combined_features) != EXPECTED_FEATURE_LENGTH:
            print(f"Warning: Feature length mismatch for {image_path}. Expected {EXPECTED_FEATURE_LENGTH}, got {len(combined_features)}. Padding/truncating.", file=sys.stderr)
            # Pad or truncate features to the expected length
            if len(combined_features) > EXPECTED_FEATURE_LENGTH:
                combined_features = combined_features[:EXPECTED_FEATURE_LENGTH]
            else:
                combined_features = np.pad(combined_features, (0, EXPECTED_FEATURE_LENGTH - len(combined_features)), 'constant')


        # --- Image Preprocessing ---
        img_resized = cv2.resize(image, target_size)
        # Convert BGR to RGB as ResNet was trained on RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # Preprocess using ResNet50's function (handles scaling etc.)
        img_preprocessed = resnet_preprocess(img_rgb) # No need for expand_dims here

        return img_preprocessed.astype(np.float32), combined_features

    except Exception as e:
        print(f"ERROR processing {image_path}: {e}", file=sys.stderr)
        return None, None

# ------------- TFLite Inference ----------------

def run_tflite_inference(tflite_model_path, test_files, test_labels):
    """Runs inference using the TFLite model on the test set."""
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
    except ValueError as e:
        print(f"ERROR: Failed to load TFLite model at {tflite_model_path}. Ensure the path is correct and the model is valid. Error: {e}")
        return
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading the TFLite model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Assuming the model has two inputs based on the notebook structure:
    # Input 0: Image (e.g., shape [1, 224, 224, 3], dtype float32)
    # Input 1: Features (e.g., shape [1, 20], dtype float32)
    # Verify this assumption:
    if len(input_details) != 2:
        print(f"ERROR: Expected TFLite model to have 2 inputs, but found {len(input_details)}.")
        print("Input details:", input_details)
        return

    # Try to identify which input is image and which is features based on shape
    # This is heuristic and might need adjustment if shapes are ambiguous
    image_input_index = -1
    feature_input_index = -1
    if len(input_details[0]['shape']) == 4 and input_details[0]['shape'][1] == IMG_SIZE:
        image_input_index = input_details[0]['index']
        feature_input_index = input_details[1]['index']
    elif len(input_details[1]['shape']) == 4 and input_details[1]['shape'][1] == IMG_SIZE:
        image_input_index = input_details[1]['index']
        feature_input_index = input_details[0]['index']
    else:
        print("ERROR: Could not reliably determine which input is for image and which for features based on shape.")
        print("Input details:", input_details)
        return

    print(f"Identified Image Input Index: {image_input_index}, Feature Input Index: {feature_input_index}")

    image_input_shape = input_details[image_input_index]['shape']
    feature_input_shape = input_details[feature_input_index]['shape']
    output_shape = output_details[0]['shape']

    print(f"Model Input Shapes: Image={image_input_shape}, Features={feature_input_shape}")
    print(f"Model Output Shape: {output_shape}")


    test_preds = []
    true_labels = []
    processed_count = 0

    for file_path, label in zip(test_files, test_labels):
        img_data, feature_data = preprocess_image_for_tflite(file_path)

        if img_data is None or feature_data is None:
            continue # Skip files that failed preprocessing

        # Add batch dimension (expected shape [1, H, W, C] for image, [1, F] for features)
        img_batch = np.expand_dims(img_data, axis=0)
        feature_batch = np.expand_dims(feature_data, axis=0)

        # Ensure data types match model input types (usually float32)
        img_batch = img_batch.astype(input_details[image_input_index]['dtype'])
        feature_batch = feature_batch.astype(input_details[feature_input_index]['dtype'])

        # Check shapes match model input shapes
        if not np.array_equal(img_batch.shape, image_input_shape):
             print(f"Warning: Image batch shape mismatch for {file_path}. Expected {image_input_shape}, got {img_batch.shape}. Skipping.", file=sys.stderr)
             continue
        if not np.array_equal(feature_batch.shape, feature_input_shape):
             print(f"Warning: Feature batch shape mismatch for {file_path}. Expected {feature_input_shape}, got {feature_batch.shape}. Skipping.", file=sys.stderr)
             continue


        try:
            # Set the tensors
            interpreter.set_tensor(image_input_index, img_batch)
            interpreter.set_tensor(feature_input_index, feature_batch)

            # Run inference
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_label = np.argmax(output_data, axis=1)[0] # Get class index with highest probability

            test_preds.append(pred_label)
            true_labels.append(label)
            processed_count += 1

        except Exception as e:
            print(f"ERROR during inference for file {file_path}: {e}", file=sys.stderr)

    if not true_labels:
        print("ERROR: No files were successfully processed for inference.")
        return

    # Calculate metrics (assuming Real=1, Fake=0 based on load_data)
    # Adjust pos_label if your convention is different (e.g., Fake=1)
    acc = accuracy_score(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds, pos_label=0) # F1 score for detecting 'Fake' (label 0)

    print(f"\n--- TFLite Inference Results ---")
    print(f"Processed {processed_count}/{len(test_files)} test files.")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score (for Fake class): {f1:.4f}")
    print(f"True Labels Sample: {true_labels[:10]}")
    print(f"Predicted Labels Sample: {test_preds[:10]}")
    print("---------------------------------")


# ------------- Main Execution ----------------

def main():
    print("Starting TFLite image inference script...")
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"ERROR: TFLite model not found at {TFLITE_MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_DATASET_DIR):
        print(f"ERROR: Image dataset directory not found at {IMAGE_DATASET_DIR}")
        return

    test_files, test_labels = load_data()

    if test_files is None or test_labels is None or not test_files:
        print("ERROR: Failed to load or split data. Exiting.")
        return

    print(f"Loaded {len(test_files)} test files for inference.")

    run_tflite_inference(TFLITE_MODEL_PATH, test_files, test_labels)

    print("Script finished.")


if __name__ == "__main__":
    main()
