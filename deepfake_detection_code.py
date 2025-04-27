# -*- coding: utf-8 -*-
"""
Deepfake Detection using a Custom CNN and Handcrafted Features.

This script downloads a dataset, extracts features (LBP, OG, SSEE) and
image embeddings (Custom CNN), prepares tf.data datasets, defines a
multi-input model, and trains it.
"""

import kagglehub
import os
import numpy as np # Used for checking dataset lengths
import math      # Used for pi in OG calculation
from datetime import datetime

# Configure TensorFlow logging (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress INFO and WARNING messages
import tensorflow as tf

# Import specific Keras components
# REMOVE ResNet50 and its preprocess_input
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
# ADD Conv2D, MaxPooling2D
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
        # Use float16 for computations where possible for speed and memory benefits
        # Requires NVIDIA GPU with Tensor Cores (Volta, Turing, Ampere+)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set to: {policy.name}")
        # --- End Mixed Precision ---
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"ERROR setting memory growth: {e}")
else:
    print("WARNING: No GPU detected. Running on CPU.")


# ----------- Feature extraction (TF) -----------
print("\n--- Defining Feature Extraction Functions ---")

@tf.function
def extract_lbp_features_tf(gray, radius=1, n_points=8):
    """
    Extracts Local Binary Pattern (LBP) features using TensorFlow operations.
    Note: This implementation extracts patches and compares neighbors to the center.
          The number of bins in the histogram is 2**n_points.

    Args:
        gray: A tf.Tensor representing the grayscale image (H, W, 1) or (H, W).
              Should be tf.uint8 or tf.float32.
        radius: The radius for the LBP neighborhood (default: 1).
        n_points: The number of points in the LBP neighborhood (default: 8).

    Returns:
        A tf.Tensor representing the normalized LBP histogram (float32, shape [2**n_points]).
    """
    gray = tf.cast(tf.squeeze(gray), tf.float32) # Ensure float32 and remove channel dim if present
    k = 2 * radius + 1 # Kernel size
    # Extract patches around each pixel
    patches = tf.image.extract_patches(
        images=tf.expand_dims(tf.expand_dims(gray, 0), -1), # Add batch and channel dims: [1, H, W, 1]
        sizes=[1, k, k, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME'
    )                # Output shape: [1, H, W, k*k]
    patches = tf.reshape(patches, [-1, k*k])    # Reshape to [H*W, k*k]
    center_idx = radius * k + radius # Index of the center pixel in the flattened patch
    center = patches[:, center_idx:center_idx+1]  # Extract center pixel values [HW, 1]
    # Extract neighbors (simplified: takes first n_points excluding center)
    neigh = tf.concat([patches[:, :center_idx], patches[:, center_idx+1:]], axis=1) # Exclude center
    neigh = neigh[:, :n_points]                  # Select n_points neighbors [HW, n_points]
    # Compare neighbors to center and create binary pattern
    bits = tf.cast(neigh >= center, tf.int32)
    # Calculate LBP codes
    weights = 2 ** tf.range(n_points, dtype=tf.int32) # Powers of 2: [1, 2, 4, ..., 2^(n_points-1)]
    codes = tf.reduce_sum(bits * weights, axis=1)  # Calculate decimal code for each pixel [HW]
    # Calculate histogram of codes
    # Using 2**n_points bins for standard LBP
    hist_size = 2**n_points
    hist = tf.cast(tf.math.bincount(codes, minlength=hist_size, maxlength=hist_size), tf.float32)
    total = tf.reduce_sum(hist)
    # Normalize histogram
    return tf.math.divide_no_nan(hist, total) # Shape [2**n_points] = [256]


@tf.function
def extract_og_features_tf(gray, n_bins=8):
    """
    Extracts Orientation Gradients (OG) histogram features using TensorFlow.

    Args:
        gray: A tf.Tensor representing the grayscale image (H, W, 1) or (H, W).
              Should be tf.uint8 or tf.float32.
        n_bins: The number of orientation bins for the histogram (default: 8).

    Returns:
        A tf.Tensor representing the normalized OG histogram (float32, shape [n_bins]).
    """
    gray = tf.cast(tf.squeeze(gray), tf.float32) # Ensure float32 and remove channel dim if present
    img = tf.expand_dims(tf.expand_dims(gray, 0), -1)  # Add batch and channel dims: [1, H, W, 1]
    # Calculate Sobel edges for gradients
    edges = tf.image.sobel_edges(img)                   # Output shape: [1, H, W, 1, 2] (y_grad, x_grad)
    gy = edges[0, ..., 0] # Gradient in y-direction [H, W, 1]
    gx = edges[0, ..., 1] # Gradient in x-direction [H, W, 1]
    # Calculate magnitude and angle
    mag = tf.sqrt(gx**2 + gy**2)
    pi_const = tf.constant(math.pi, dtype=tf.float32) # Use math.pi
    ang = tf.atan2(gy, gx) * (180. / pi_const)
    ang = tf.where(ang < 0, ang + 360, ang) # Convert angles to [0, 360] range
    # Bin angles
    bin_size = 360. / tf.cast(n_bins, tf.float32)
    idx = tf.cast(ang / bin_size, tf.int32)
    idx = tf.clip_by_value(idx, 0, n_bins-1) # Ensure indices are within [0, n_bins-1]
    # Create histogram weighted by magnitude
    mag_flat = tf.reshape(mag, [-1]) # Flatten magnitude tensor
    idx_flat = tf.reshape(idx, [-1]) # Flatten index tensor
    # Sum magnitudes for each bin index
    sums = tf.math.unsorted_segment_sum(mag_flat, idx_flat, n_bins)
    total = tf.reduce_sum(sums)
    # Normalize histogram
    return tf.math.divide_no_nan(sums, total) # Shape [n_bins] = [8]


@tf.function
def extract_ssee_features_tf(img, gray, quality=75):
    """
    Extracts Steganalysis Statistical Error Estimation (SSEE) features using TensorFlow.
    Calculates the mean and standard deviation of the difference between the original
    grayscale image and a re-compressed JPEG version.

    Args:
        img: The original color image tensor [H, W, 3], tf.uint8 or tf.float32.
        gray: The original grayscale image tensor [H, W, 1] or [H, W], tf.uint8 or tf.float32.
        quality: The JPEG compression quality (0-100, default: 75).

    Returns:
        A tf.Tensor containing [mean_difference, std_dev_difference] (float32, shape [2]).
    """
    img_u8 = tf.cast(img, tf.uint8) # Ensure uint8 for JPEG encoding
    gray_u8 = tf.cast(tf.squeeze(gray), tf.uint8) # Ensure uint8 and remove channel dim
    # Encode the color image as JPEG
    jpeg = tf.io.encode_jpeg(img_u8, quality=quality)
    # Decode the JPEG back to grayscale
    dec = tf.io.decode_jpeg(jpeg, channels=1) # Decode directly to grayscale [H, W, 1]
    # Calculate absolute difference between original gray and decoded gray
    diff = tf.abs(tf.cast(gray_u8, tf.float32) - tf.cast(dec[...,0], tf.float32))
    # Calculate mean and standard deviation of the difference
    m = tf.reduce_mean(diff)
    s = tf.math.reduce_std(diff)
    return tf.stack([m, s], axis=0) # Shape [2]


# Combine feature extraction and image preprocessing
@tf.function # Uncommented for potential performance boost
def extract_combined_features_and_image_tf(path):
    """
    Reads an image file, extracts handcrafted features (LBP, OG, SSEE),
    resizes and preprocesses the image for the custom CNN.

    Args:
        path: A scalar tf.string tensor representing the image file path.

    Returns:
        A tuple containing:
        - img_pre: Preprocessed image tensor [IMG_SIZE, IMG_SIZE, 3] (float32) for the CNN.
        - feats: Combined handcrafted features tensor [EXPECTED_FEATURE_LENGTH] (float32).
    """
    try:
        raw = tf.io.read_file(path)
        # Decode image, ensure 3 channels (RGB)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        # Set shape if unknown after decoding (important for subsequent ops)
        img.set_shape([None, None, 3])
        img_float = tf.cast(img, tf.float32) # Cast to float for feature extraction

        # Convert to grayscale for LBP and OG
        # Use tf.image.rgb_to_grayscale which returns shape [H, W, 1]
        gray = tf.image.rgb_to_grayscale(img_float) # Keep channel dim for consistency

        # Extract handcrafted features
        lbp_feats = extract_lbp_features_tf(gray, radius=1, n_points=8) # Output shape: [256]
        og_feats = extract_og_features_tf(gray, n_bins=8)              # Output shape: [8]
        ssee_feats = extract_ssee_features_tf(img_float, gray, quality=75)   # Output shape: [2]

        # Concatenate features
        feats = tf.concat([lbp_feats, og_feats, ssee_feats], axis=0)
        # Ensure feature length matches expectation
        tf.debugging.assert_equal(tf.shape(feats)[0], EXPECTED_FEATURE_LENGTH,
                                  message=f"Feature length mismatch for {path}")


        # Preprocess image for the custom CNN model
        # Use tf.image.resize with the global IMG_SIZE
        img_resized = tf.image.resize(img_float, [IMG_SIZE, IMG_SIZE]) # Resize to CNN input size
        # Simple rescaling to [0, 1] - common for custom CNNs
        img_pre = img_resized / 255.0
        # REMOVE ResNet-specific preprocessing
        # img_pre = resnet_preprocess(img_resized)

        return img_pre, feats

    except Exception as e:
        # Handle potential errors during file reading or processing
        # Use tf.print for messages inside tf.function or tf.data pipelines
        tf.print("Error processing image:", path, "Error:", e, output_stream=tf.sys.stderr)
        # Return dummy data with correct shapes to allow dataset processing to continue
        # Errors can be filtered out later if needed, or training might fail if too many errors
        # Ensure dummy data types match expected output types (float32)
        dummy_img = tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
        dummy_feats = tf.zeros([EXPECTED_FEATURE_LENGTH], dtype=tf.float32)
        return dummy_img, dummy_feats


# ----------- Dataset Preparation ------------
print("\n--- Preparing Datasets ---")

def list_image_paths_and_labels(directory):
    """
    Lists image paths and corresponding labels (0 for Fake, 1 for Real)
    within a given directory structure (e.g., directory/Fake, directory/Real).

    Args:
        directory: The base directory containing 'Fake' and 'Real' subdirectories.

    Returns:
        A tuple containing:
        - paths: A list of full image file paths.
        - labels: A list of corresponding integer labels (0 or 1).
    """
    cats = ['Fake', 'Real'] # Assuming 'Fake' is class 0, 'Real' is class 1
    paths, labels = [], []
    print(f"Scanning directory: {directory}")
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found - {directory}")
        return paths, labels # Return empty lists

    for idx, cat in enumerate(cats):
        cat_dir = os.path.join(directory, cat)
        if not os.path.isdir(cat_dir):
            print(f"Warning: Sub-directory not found - {cat_dir}")
            continue
        print(f"  Scanning sub-directory: {cat_dir}")
        count = 0
        for f in os.listdir(cat_dir):
            # Check for common image file extensions
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff','.gif')):
                paths.append(os.path.join(cat_dir, f))
                labels.append(idx) # Assign label based on subdirectory index
                count += 1
            # else:
            #     print(f"Skipping non-image file: {f}") # Optional: log skipped files
        print(f"    Found {count} images in {cat}.")
    return paths, labels


def create_dataset(image_paths, labels, batch_size=32, shuffle=True, dataset_name=""):
    """
    Creates a tf.data.Dataset for image classification with combined features.
    Yields dictionaries mapping input names to tensors.

    Args:
        image_paths: A list of image file paths.
        labels: A list of corresponding integer labels.
        batch_size: The batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        dataset_name: A string name for logging purposes (e.g., "Training").

    Returns:
        A tf.data.Dataset yielding ({'image_input': image, 'feature_input': features}, one_hot_label) tuples.
        Returns None if image_paths is empty.
    """
    if not image_paths:
        print(f"Warning: No image paths provided for {dataset_name} dataset. Returning None.")
        return None # Or return an empty dataset: tf.data.Dataset.from_tensor_slices(([], []))
    if len(image_paths) != len(labels):
         print(f"ERROR: Mismatch between number of paths ({len(image_paths)}) and labels ({len(labels)}) for {dataset_name} dataset.")
         return None


    print(f"Creating {dataset_name} dataset with {len(image_paths)} samples...")
    AUTOTUNE = tf.data.AUTOTUNE # Enable optimal parallel processing

    # --- THIS IS THE KEY FIX ---
    # The wrapper function now returns a dictionary for the inputs,
    # matching the names given to the Input layers in the Keras model.
    def _wrapper(path, lbl):
        img_pre, feats = extract_combined_features_and_image_tf(path)
        onehot = tf.one_hot(lbl, depth=NUM_CLASSES) # Use global NUM_CLASSES
        # Return a dictionary for the model's inputs
        inputs = {
            'image_input': img_pre,    # Key matches the name of the image Input layer
            'feature_input': feats     # Key matches the name of the feature Input layer
        }
        return inputs, onehot
    # --- END OF FIX ---

    # Create dataset from slices of paths and labels
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        # Shuffle the dataset - use buffer size equal to dataset size for full shuffle
        # Use reshuffle_each_iteration=True for better randomness across epochs
        print(f"  Shuffling {dataset_name} dataset (buffer size: {len(image_paths)})...")
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    # Map the processing function in parallel
    # num_parallel_calls=AUTOTUNE lets TensorFlow decide the parallelism level
    # deterministic=False can improve performance but might change order slightly if shuffling
    print(f"  Mapping processing function for {dataset_name} dataset...")
    # Set deterministic=False only when shuffle=True
    ds = ds.map(_wrapper, num_parallel_calls=AUTOTUNE, deterministic=not shuffle)

    # --- Add Caching ---
    # Cache the dataset after mapping. If the dataset fits in memory, this
    # significantly speeds up subsequent epochs by avoiding repeated preprocessing.
    # If it doesn't fit in memory, TF will cache to a local file.
    print(f"  Caching {dataset_name} dataset...")
    ds = ds.cache()
    # --- End Caching ---

    # Batch the dataset
    print(f"  Batching {dataset_name} dataset (batch size: {batch_size})...")
    ds = ds.batch(batch_size)

    # Prefetch data for better performance (allows data loading while GPU is training)
    print(f"  Prefetching {dataset_name} dataset...")
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    print(f"{dataset_name} dataset created successfully.")
    return ds


# List image paths and labels for each split
train_paths, train_labels = list_image_paths_and_labels(train_dir)
test_paths,  test_labels  = list_image_paths_and_labels(test_dir)
val_paths,   val_labels   = list_image_paths_and_labels(val_dir)

print(f"\nFound {len(train_paths)} training images.")
print(f"Found {len(test_paths)} testing images.")
print(f"Found {len(val_paths)} validation images.")

# Create the datasets using the corrected function
train_ds = create_dataset(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True, dataset_name="Training")
val_ds   = create_dataset(val_paths,   val_labels,   batch_size=BATCH_SIZE, shuffle=False, dataset_name="Validation")
test_ds  = create_dataset(test_paths,  test_labels,  batch_size=BATCH_SIZE, shuffle=False, dataset_name="Testing")


# Verify dataset creation and check element spec
if train_ds and val_ds and test_ds:
    print("\nDatasets ready!")
    print("Example element spec (Train DS):", train_ds.element_spec)
    # ((Dict{'feature_input': TensorSpec(shape=(None, 20), dtype=tf.float32, name=None),
    #        'image_input': TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None)},
    # TensorSpec(shape=(None, 2), dtype=tf.float32, name=None))

    # Optional: Iterate over one batch to check shapes and trigger potential errors early
    print("\nChecking one batch from training dataset...")
    try:
        # Get one batch from the training dataset
        for inputs_batch, labels_batch in train_ds.take(1):
            # Access inputs using the dictionary keys
            images = inputs_batch['image_input']
            features = inputs_batch['feature_input']
            print(f"  Images batch shape: {images.shape}")      # Should be (BATCH_SIZE, 224, 224, 3)
            print(f"  Features batch shape: {features.shape}")    # Should be (BATCH_SIZE, 20)
            print(f"  Labels batch shape: {labels_batch.shape}")    # Should be (BATCH_SIZE, 2)
        print("Batch check successful.")
    except Exception as e:
        print(f"ERROR during batch check: {e}")
        print("There might be an issue with data loading or preprocessing for some files.")
        # Consider adding more robust error handling or filtering in the dataset pipeline
        exit() # Exit if the batch check fails

else:
    print("\nERROR: One or more datasets could not be created. Check previous warnings/errors.")
    exit() # Exit if datasets aren't ready


# ----------- Model Definition ------------
print("\n--- Building Multi-Input Model ---")

# --- Image Input Branch (Custom CNN) ---
# Use the globally defined IMG_SIZE
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input') # Name matches dataset key

# Define a smaller CNN base model
x = Conv2D(16, (3, 3), activation='relu', padding='same')(image_input) # Reduced filters
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) # Reduced filters
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

# Optional: Remove this block for an even smaller model
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # Reduced filters
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)

# Add a global average pooling layer to flatten the features
image_features = GlobalAveragePooling2D(name='image_gap')(x) # Output shape: (None, 32) if last Conv2D has 32 filters
# Optional: Add a Dropout layer after pooling
# image_features = Dropout(0.5)(image_features) # Example dropout

# --- Handcrafted Features Input Branch ---
# Use the globally defined EXPECTED_FEATURE_LENGTH
feature_input = Input(shape=(EXPECTED_FEATURE_LENGTH,), name='feature_input') # Name matches dataset key
# Add some Dense layers to process the handcrafted features
feature_dense = Dense(64, activation='relu', name='feature_dense_1')(feature_input)
feature_dense = Dense(32, activation='relu', name='feature_dense_2')(feature_dense) # Output shape: (None, 32)

# --- Combine Branches ---
# Concatenate the output of the image branch and the feature branch
combined = concatenate([image_features, feature_dense], name='concatenate_features') # Shape: (None, 32 + 32) if last Conv2D has 32 filters

# --- Classification Head ---
# Add final Dense layers for classification
final_dense = Dense(128, activation='relu', name='final_dense_1')(combined) # Adjusted size based on smaller combined features
# Optional: Add Dropout before the final layer
# final_dense = Dropout(0.5)(final_dense)
# Output layer with NUM_CLASSES units and softmax activation for probabilities
# Ensure the final layer uses float32 for numerical stability with mixed precision
predictions = Dense(NUM_CLASSES, activation='softmax', name='output_predictions', dtype='float32')(final_dense)

# --- Create Model ---
# Define the model with two inputs and one output
multi_input_model = Model(inputs=[image_input, feature_input], outputs=predictions)

# --- Compile Model ---
multi_input_model.compile(
    optimizer=Adam(learning_rate=0.0001), # May need adjustment for custom CNN
    loss='categorical_crossentropy',      # Suitable for one-hot encoded labels
    metrics=['accuracy'],                 # Track accuracy during training
    jit_compile=True                      # Keep XLA if desired
)

print("Multi-input model built and compiled successfully.")
print("Model Summary:")
multi_input_model.summary(print_fn=lambda x: print(x))


# ----------- Training ------------
print("\n--- Setting up Training Callbacks ---")

# Model Checkpoint: Save the best model based on validation loss
checkpoint_dir = './checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'best_model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras')
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',       # Monitor validation loss
    save_best_only=True,      # Only save if val_loss improves
    save_weights_only=False,  # Save the full model
    mode='min',               # Minimize the monitored quantity (loss)
    verbose=1
)

# TensorBoard Callback: Log training progress for visualization
log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,      # Log histogram visualizations every epoch
    write_graph=True,      # Visualize the model graph in TensorBoard
    write_images=False,    # Set to True to log model weights as images (can be large)
    update_freq='epoch'    # Log metrics after each epoch
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
    train_ds,                   # Training data (yields ({inputs}, labels))
    epochs=EPOCHS,
    validation_data=val_ds,     # Validation data (yields ({inputs}, labels))
    callbacks=[checkpoint_callback, tensorboard_callback], # List of callbacks
    verbose=1                   # Show progress bar
)

print("\n--- Model Training Complete ---")
print(f"Training history keys: {history.history.keys()}")
# You can now analyze the 'history' object (e.g., plot loss/accuracy)

# --- Optional: Evaluate on Test Set ---
if test_ds:
    print("\n--- Evaluating Model on Test Set ---")
    test_loss, test_accuracy = multi_input_model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
else:
    print("\nTest dataset not available, skipping evaluation.")

print("\n--- Script Finished ---")