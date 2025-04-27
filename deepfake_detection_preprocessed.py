# -*- coding: utf-8 -*-
"""
Deepfake Detection using a Custom CNN and Handcrafted Features.

This script first preprocesses the entire dataset (extracts features, resizes images)
using OpenCV and Scikit-image, saves it to TFRecord files. Then, it loads the
preprocessed data using tf.data.TFRecordDataset, defines a multi-input model,
and trains it.
"""

import kagglehub
import os
import numpy as np # Used for checking dataset lengths and feature extraction
import math      # Used for pi in OG calculation (now potentially unused)
from datetime import datetime
import time # For timing preprocessing
import cv2 # Import OpenCV
from skimage.feature import local_binary_pattern # Import LBP from scikit-image

# Configure TensorFlow logging (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
import tensorflow as tf

# Import specific Keras components
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import mixed_precision # Import mixed precision

# -------- Constants ----------
IMG_SIZE = 224       # Input image size
NUM_CLASSES = 2      # Number of output classes (Fake, Real)
# Expected feature length: LBP (2^8=256) + OG (8) + SSEE (2) = 266
EXPECTED_FEATURE_LENGTH = 266
BATCH_SIZE = 64      # Adjust based on GPU memory
PREPROCESSED_DATA_DIR = './preprocessed_data' # Directory to save TFRecords

# -------- Download Dataset ----------
print("--- Downloading Dataset ---")
try:
    # Ensure you are authenticated with Kaggle if running locally: kagglehub.login()
    # Use kagglehub to download the dataset and get the path
    dataset_path = kagglehub.dataset_download('manjilkarki/deepfake-and-real-images')
    print(f"Dataset downloaded to: {dataset_path}")
    # Construct the base directory path for the 'Dataset' folder within the downloaded path
    base_dir = os.path.join(dataset_path, 'Dataset')
    # Verify the base directory exists
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"ERROR: The expected 'Dataset' directory was not found inside {dataset_path}")
    print('Data source import complete.')

    # Define data directories
    train_dir = os.path.join(base_dir, 'Train')
    test_dir = os.path.join(base_dir, 'Test')
    val_dir = os.path.join(base_dir, 'Validation')

    print(f"Using training data from: {train_dir}")
    print(f"Using testing data from: {test_dir}")
    print(f"Using validation data from: {val_dir}")

except Exception as e:
    print(f"ERROR downloading or finding dataset: {e}")
    exit() # Exit if dataset is not available

# -------- GPU SETUP ----------
print("\n--- GPU Setup ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable dynamic memory growth for the first GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Using GPU: {gpus[0].name}")
        print("Memory growth enabled.")
        # --- Enable Mixed Precision ---
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set to: {policy.name}")
        # --- End Mixed Precision ---
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"ERROR setting memory growth: {e}")
else:
    print("WARNING: No GPU detected. Running on CPU.")


# ----------- Feature extraction (OpenCV/Scikit-image) - Used for preprocessing -----------
print("\n--- Defining Feature Extraction Functions (OpenCV/Scikit-image) ---")

def extract_lbp_features(gray_image, radius=1, n_points=8):
    """Extracts LBP features using scikit-image."""
    try:
        # Use default method (not 'uniform') to get 2^n_points bins
        lbp = local_binary_pattern(gray_image, n_points, radius, method='default')
        n_bins = int(2**n_points) # Expect 256 bins for n_points=8
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        # Ensure histogram has the correct length, even if some bins are empty
        if len(hist) != n_bins:
             print(f"Warning: LBP histogram length mismatch. Expected {n_bins}, got {len(hist)}. Padding with zeros.")
             # Pad if necessary, although density=True should handle the range correctly
             hist = np.pad(hist, (0, n_bins - len(hist)), 'constant')
        return hist.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Error processing LBP: {e}")
        return np.zeros(2**n_points, dtype=np.float32) # Return array of expected size

def extract_og_features(gray_image, n_bins=8):
    """Extracts OG features using OpenCV."""
    try:
        gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(angle.ravel(), bins=n_bins, range=(0, 360), weights=magnitude.ravel(), density=False)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist = hist / hist_sum # Normalize
        else:
            # Handle case where magnitude sum is zero (e.g., black image)
            hist = np.zeros(n_bins)
        return hist.astype(np.float32)
    except Exception as e:
        print(f"ERROR: Error in OG extraction: {e}")
        return np.zeros(n_bins, dtype=np.float32)

def extract_ssee_features(image, gray_image, quality=75):
    """Extracts SSEE features using OpenCV."""
    try:
        # Ensure gray_image is 8-bit for comparison
        if gray_image.dtype != np.uint8:
            gray_image_u8 = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            gray_image_u8 = gray_image

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', image, encode_param) # Encode original color image
        if not result:
            print("Warning: Failed to encode image for SSEE.")
            return np.array([0.0, 0.0], dtype=np.float32)

        # Decode the compressed image as grayscale
        decimg_gray = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        if decimg_gray is None:
            print("Warning: Failed to decode image for SSEE.")
            return np.array([0.0, 0.0], dtype=np.float32)

        # Ensure shapes match before diff (resize decoded if necessary)
        if decimg_gray.shape != gray_image_u8.shape:
            # print(f"Warning: Resizing decoded JPEG ({decimg_gray.shape}) to match original gray ({gray_image_u8.shape}) for SSEE.")
            decimg_gray = cv2.resize(decimg_gray, (gray_image_u8.shape[1], gray_image_u8.shape[0]))

        # Calculate absolute difference
        diff = cv2.absdiff(gray_image_u8, decimg_gray)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        return np.array([mean_diff, std_diff], dtype=np.float32)
    except Exception as e:
        print(f"ERROR: Error in SSEE (JPEG Error) extraction: {e}")
        return np.array([0.0, 0.0], dtype=np.float32)

def extract_combined_features_and_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Reads an image file using OpenCV, extracts features, preprocesses image.
    Returns None, None on error.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return None, None

        # Convert to grayscale for feature extraction
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract features
        lbp_hist = extract_lbp_features(gray_image)
        og_hist = extract_og_features(gray_image)
        ssee_feats = extract_ssee_features(image, gray_image) # Pass color image for JPEG encoding

        # Combine features
        combined_features = np.concatenate((lbp_hist, og_hist, ssee_feats)).astype(np.float32)

        # Check feature length
        if len(combined_features) != EXPECTED_FEATURE_LENGTH:
            print(f"Warning: Feature length mismatch for {image_path}. Expected {EXPECTED_FEATURE_LENGTH}, got {len(combined_features)}. Skipping.")
            # print(f"  LBP: {len(lbp_hist)}, OG: {len(og_hist)}, SSEE: {len(ssee_feats)}") # Debugging lengths
            return None, None

        # Resize and preprocess image (convert BGR to RGB and scale)
        img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) # Convert to RGB
        img_preprocessed = img_rgb.astype(np.float32) / 255.0 # Scale to [0, 1]

        # Ensure shapes are correct
        img_preprocessed = np.reshape(img_preprocessed, (IMG_SIZE, IMG_SIZE, 3))
        combined_features = np.reshape(combined_features, (EXPECTED_FEATURE_LENGTH,))

        return img_preprocessed, combined_features

    except Exception as e:
        print(f"ERROR processing {image_path}: {e}")
        return None, None


# ----------- TFRecord Helper Functions -----------
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  # Ensure value is a list or similar structure
  if not isinstance(value, (list, np.ndarray, tf.Tensor)):
      value = [value]
  # If it's a tensor, convert to numpy list
  if isinstance(value, tf.Tensor):
      value = value.numpy().tolist()
  # If it's a numpy array, convert to list
  elif isinstance(value, np.ndarray):
      value = value.tolist()
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, features, label):
  """
  Creates a tf.train.Example message ready to be written to a file.
  Accepts numpy arrays for image and features.
  """
  # Convert numpy arrays to TensorFlow tensors before serialization if needed,
  # or serialize directly from numpy bytes. Using tf.io.serialize_tensor is convenient.
  image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
  features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)

  image_bytes = tf.io.serialize_tensor(image_tensor) # Serialize the preprocessed image tensor

  feature = {
      'image_raw': _bytes_feature(image_bytes), # Store serialized tensor
      'features': _float_feature(features_tensor), # Store feature vector (already float32)
      'label': _int64_feature(label),           # Store integer label
  }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


# ----------- Preprocessing and Saving to TFRecord -----------
print("\n--- Preprocessing Data and Saving to TFRecords ---")
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

def preprocess_and_save_tfrecord(image_paths, labels, output_filename, dataset_name=""):
    """
    Processes images and features using OpenCV/Skimage, then saves them to a TFRecord file.
    """
    if not image_paths:
        print(f"Warning: No image paths provided for {dataset_name}. Skipping TFRecord creation.")
        return 0 # Return 0 samples processed

    output_filepath = os.path.join(PREPROCESSED_DATA_DIR, output_filename)
    print(f"Starting preprocessing for {dataset_name} dataset ({len(image_paths)} images)...")
    print(f"Output TFRecord file: {output_filepath}")
    start_time = time.time()
    samples_processed = 0
    samples_skipped = 0

    with tf.io.TFRecordWriter(output_filepath) as writer:
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            if (i + 1) % 100 == 0:
                print(f"  {dataset_name}: Processed {i+1}/{len(image_paths)} images...")

            # Use the OpenCV/Skimage based extraction function
            image_preprocessed, features_extracted = extract_combined_features_and_image(path)

            # Check if processing was successful
            if image_preprocessed is not None and features_extracted is not None:
                # Serialize and write to TFRecord
                # Ensure data types are correct before serialization if necessary
                # serialize_example handles tensor conversion
                example = serialize_example(image_preprocessed, features_extracted, label)
                writer.write(example)
                samples_processed += 1
            else:
                # Error/skip message is printed within extract_combined_features_and_image
                samples_skipped += 1 # Count skipped images

    end_time = time.time()
    print(f"Finished preprocessing {dataset_name} dataset.")
    print(f"  Successfully processed and saved: {samples_processed} samples.")
    print(f"  Skipped due to errors: {samples_skipped} samples.")
    print(f"  Time taken: {end_time - start_time:.2f} seconds.")
    return samples_processed # Return the number of samples actually saved


# ------------- Dataset Listing Function ----------------
def list_image_paths_and_labels(base_dir):
    """Lists image file paths and assigns labels based on subdirectories."""
    categories = ['Fake', 'Real'] # Assuming these are the subfolder names
    paths = []
    labels = []
    print(f"Listing images in: {base_dir}")
    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found - {base_dir}")
        return paths, labels

    for label, category in enumerate(categories):
        category_dir = os.path.join(base_dir, category)
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found - {category_dir}")
            continue
        print(f"  Processing category: {category} (Label: {label})")
        found_files = 0
        for filename in os.listdir(category_dir):
            # Check for common image file extensions
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                paths.append(os.path.join(category_dir, filename))
                labels.append(label)
                found_files += 1
        print(f"    Found {found_files} images.")
    return paths, labels


# List image paths and labels for each split
train_paths, train_labels = list_image_paths_and_labels(train_dir)
test_paths,  test_labels  = list_image_paths_and_labels(test_dir)
val_paths,   val_labels   = list_image_paths_and_labels(val_dir)

print(f"\nFound {len(train_paths)} training images.")
print(f"Found {len(test_paths)} testing images.")
print(f"Found {len(val_paths)} validation images.")

# Preprocess and save each dataset split
# Only run preprocessing if TFRecord files don't exist or if forced
force_preprocessing = False # Set to True to always re-run preprocessing

train_tfrecord_file = os.path.join(PREPROCESSED_DATA_DIR, 'train.tfrecord')
val_tfrecord_file = os.path.join(PREPROCESSED_DATA_DIR, 'validation.tfrecord')
test_tfrecord_file = os.path.join(PREPROCESSED_DATA_DIR, 'test.tfrecord')

num_train_samples = 0
num_val_samples = 0
num_test_samples = 0

if not os.path.exists(train_tfrecord_file) or force_preprocessing:
    num_train_samples = preprocess_and_save_tfrecord(train_paths, train_labels, 'train.tfrecord', "Training")
else:
    print(f"TFRecord file already exists: {train_tfrecord_file}. Skipping preprocessing.")
    # Ideally, store the sample count somewhere or count records if needed later
    # For simplicity, we'll rely on the count from the last run or assume it matches path list length if not run.
    num_train_samples = len(train_paths) # Approximate if file exists

if not os.path.exists(val_tfrecord_file) or force_preprocessing:
    num_val_samples = preprocess_and_save_tfrecord(val_paths, val_labels, 'validation.tfrecord', "Validation")
else:
    print(f"TFRecord file already exists: {val_tfrecord_file}. Skipping preprocessing.")
    num_val_samples = len(val_paths) # Approximate

if not os.path.exists(test_tfrecord_file) or force_preprocessing:
    num_test_samples = preprocess_and_save_tfrecord(test_paths, test_labels, 'test.tfrecord', "Testing")
else:
    print(f"TFRecord file already exists: {test_tfrecord_file}. Skipping preprocessing.")
    num_test_samples = len(test_paths) # Approximate


# ----------- Dataset Preparation from TFRecords ------------
print("\n--- Preparing Datasets from TFRecords ---")

# Define the parsing function for TFRecord examples
def parse_tfrecord_fn(example_proto):
    # Define the features structure used during serialization
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string), # Serialized tensor
        'features': tf.io.FixedLenFeature([EXPECTED_FEATURE_LENGTH], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    # Parse the input tf.train.Example proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)

    # Deserialize the image tensor
    image = tf.io.parse_tensor(example['image_raw'], out_type=tf.float32)
    # Ensure shape is set correctly after parsing
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])

    features = example['features']
    label = tf.cast(example['label'], tf.int32) # Cast label to int32 if needed
    one_hot_label = tf.one_hot(label, depth=NUM_CLASSES)

    # Return the dictionary structure expected by the model
    inputs = {
        'image_input': image,
        'feature_input': features
    }
    return inputs, one_hot_label


def create_dataset_from_tfrecord(tfrecord_filename, batch_size=32, shuffle=True, dataset_name="", buffer_size=None):
    """
    Creates a tf.data.Dataset from a TFRecord file.

    Args:
        tfrecord_filename: Path to the TFRecord file.
        batch_size: Batch size.
        shuffle: Whether to shuffle the dataset.
        dataset_name: Name for logging.
        buffer_size: Shuffle buffer size. If None, uses the number of samples (requires knowing it).

    Returns:
        A tf.data.Dataset or None if the file doesn't exist.
    """
    if not os.path.exists(tfrecord_filename):
        print(f"ERROR: TFRecord file not found: {tfrecord_filename}")
        return None

    print(f"Creating {dataset_name} dataset from {tfrecord_filename}...")
    AUTOTUNE = tf.data.AUTOTUNE

    raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)

    # Optional: Add caching *before* parsing if TFRecords are read multiple times across epochs
    # raw_dataset = raw_dataset.cache() # Cache the raw serialized data

    if shuffle:
        # For true shuffling, buffer_size should ideally be the size of the dataset.
        # If buffer_size is not provided, shuffling might be limited.
        if buffer_size is None:
            print(f"Warning: Shuffle buffer size not provided for {dataset_name}. Shuffling may be limited.")
            # Estimate size or use a large fixed buffer if dataset size is unknown
            buffer_size = 10000 # Example large buffer
        print(f"  Shuffling {dataset_name} dataset (buffer size: {buffer_size})...")
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Parse the records using the parse function
    print(f"  Mapping parsing function for {dataset_name} dataset...")
    # Set deterministic=False only when shuffle=True for potential performance gain
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE, deterministic=not shuffle)

    # Optional: Cache the parsed data (images, features, labels)
    # This is useful if parsing is expensive and the parsed data fits in memory.
    print(f"  Caching parsed {dataset_name} dataset...")
    parsed_dataset = parsed_dataset.cache()

    # Batch the dataset
    print(f"  Batching {dataset_name} dataset (batch size: {batch_size})...")
    ds = parsed_dataset.batch(batch_size)

    # Prefetch data
    print(f"  Prefetching {dataset_name} dataset...")
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    print(f"{dataset_name} dataset created successfully.")
    return ds


# Create datasets from the TFRecord files
# Pass the approximate number of samples for better shuffling if available
train_ds = create_dataset_from_tfrecord(train_tfrecord_file, batch_size=BATCH_SIZE, shuffle=True, dataset_name="Training", buffer_size=num_train_samples if num_train_samples > 0 else None)
val_ds   = create_dataset_from_tfrecord(val_tfrecord_file,   batch_size=BATCH_SIZE, shuffle=False, dataset_name="Validation") # No shuffle or buffer needed for validation
test_ds  = create_dataset_from_tfrecord(test_tfrecord_file,  batch_size=BATCH_SIZE, shuffle=False, dataset_name="Testing")   # No shuffle or buffer needed for testing


# Verify dataset creation and check element spec
if train_ds and val_ds and test_ds:
    print("\nDatasets ready!")
    print("Example element spec (Train DS):", train_ds.element_spec)
    # Expected output structure:
    # ({'image_input': TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None),
    #   'feature_input': TensorSpec(shape=(None, 266), dtype=tf.float32, name=None)},
    #  TensorSpec(shape=(None, 2), dtype=tf.float32, name=None))

    # Optional: Iterate over one batch to check shapes and trigger potential errors early
    print("\nChecking one batch from training dataset...")
    try:
        for inputs_batch, labels_batch in train_ds.take(1):
            images = inputs_batch['image_input']
            features = inputs_batch['feature_input']
            print(f"  Images batch shape: {images.shape}")      # Should be (BATCH_SIZE, 224, 224, 3)
            print(f"  Features batch shape: {features.shape}")    # Should be (BATCH_SIZE, 266)
            print(f"  Labels batch shape: {labels_batch.shape}")    # Should be (BATCH_SIZE, 2)
        print("Batch check successful.")
    except Exception as e:
        print(f"ERROR during batch check: {e}")
        print("There might be an issue with TFRecord parsing or the data itself.")
        exit() # Exit if the batch check fails

else:
    print("\nERROR: One or more datasets could not be created from TFRecords. Check previous warnings/errors.")
    exit() # Exit if datasets aren't ready


# ----------- Model Definition ------------
print("\n--- Building Multi-Input Model ---")

# --- Image Input Branch (Custom CNN) ---
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input') # Name matches dataset key
x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
image_features = GlobalAveragePooling2D(name='image_gap')(x)

# --- Handcrafted Features Input Branch ---
feature_input = Input(shape=(EXPECTED_FEATURE_LENGTH,), name='feature_input') # Name matches dataset key
feature_dense = Dense(64, activation='relu', name='feature_dense_1')(feature_input)
feature_dense = Dense(32, activation='relu', name='feature_dense_2')(feature_dense)

# --- Combine Branches ---
combined = concatenate([image_features, feature_dense], name='concatenate_features')

# --- Classification Head ---
final_dense = Dense(128, activation='relu', name='final_dense_1')(combined)
predictions = Dense(NUM_CLASSES, activation='softmax', name='output_predictions', dtype='float32')(final_dense)

# --- Create Model ---
multi_input_model = Model(inputs=[image_input, feature_input], outputs=predictions)

# --- Compile Model ---
multi_input_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    jit_compile=True # Keep XLA if desired
)

print("Multi-input model built and compiled successfully.")
print("Model Summary:")
multi_input_model.summary(print_fn=lambda x: print(x))


# ----------- Training ------------
print("\n--- Setting up Training Callbacks ---")

# Model Checkpoint
checkpoint_dir = './checkpoints_preprocessed' # Use a different dir if needed
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# TensorBoard Callback
log_dir = os.path.join("logs_preprocessed", "fit", datetime.now().strftime("%Y%m%d-%H%M%S")) # Use a different dir
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=False,
    update_freq='epoch'
)

print(f"Checkpoints will be saved to: {checkpoint_dir}")
print(f"TensorBoard logs will be saved to: {log_dir}")

# --- Train Model ---
print("\n--- Starting Model Training ---")

EPOCHS = 25 # Number of training epochs

# Ensure datasets are not None before starting training
if train_ds is None or val_ds is None:
    print("ERROR: Training or Validation dataset is not available. Exiting.")
    exit()

history = multi_input_model.fit(
    train_ds,                   # Training data from TFRecords
    epochs=EPOCHS,
    validation_data=val_ds,     # Validation data from TFRecords
    callbacks=[checkpoint_callback, tensorboard_callback],
    verbose=1
)

print("\n--- Model Training Complete ---")
print(f"Training history keys: {history.history.keys()}")

# --- Optional: Evaluate on Test Set ---
if test_ds:
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_accuracy = multi_input_model.evaluate(test_ds, verbose=1) # Use test_ds from TFRecords
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
else:
    print("\nTest dataset not available, skipping evaluation.")

print("\n--- Script Finished ---")
