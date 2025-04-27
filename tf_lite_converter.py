import os

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split

model_name = "final_model"  # Your Keras model name
tf_lite_model_name = "best_model"  # TFLite model name
quantization_type = "float16"  # options: "none", "dynamic", "float16", "int8"

# Load your Keras model
model = tf.keras.models.load_model(f"{model_name}.keras")

# Set up the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Constants
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
AUDIO_DIR = os.path.join(os.getcwd(), "AUDIO_DATASET")
REAL_DIR = os.path.join(AUDIO_DIR, "real_wav")
FAKE_DIR = os.path.join(AUDIO_DIR, "fake_wav")


def load_data():
    real_files = [
        os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.endswith(".wav")
    ]
    fake_files = [
        os.path.join(FAKE_DIR, f) for f in os.listdir(FAKE_DIR) if f.endswith(".wav")
    ]
    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        files, labels, test_size=0.3, random_state=42, shuffle=True
    )

    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.33, random_state=42, shuffle=True
    )

    return val_files, val_labels


def preprocess_audio(filepath, sample_rate=16000):
    y, sr = sf.read(filepath)

    if sr != sample_rate:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=sample_rate)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=64)
    log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

    log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / np.std(
        log_mel_spectrogram
    )

    # Target number of frames (columns) we want in the final log-mel spectrogram
    n_frames = 32

    # If the number of frames (i.e., width of the spectrogram) is less than 32
    if log_mel_spectrogram.shape[1] < n_frames:
        # Pad the spectrogram with zeros along the time axis (columns) to make it exactly 32 frames
        # (0, 0) => no padding on the frequency axis (rows)
        # (0, n_frames - current_frames) => pad at the end to reach 32 frames
        log_mel_spectrogram = np.pad(
            log_mel_spectrogram,
            ((0, 0), (0, n_frames - log_mel_spectrogram.shape[1])),  # padding settings
            mode="constant",  # pad with constant values (default is 0)
        )
    else:
        # If there are more than 32 frames, cut (truncate) the spectrogram to keep only the first 32 frames
        log_mel_spectrogram = log_mel_spectrogram[:, :n_frames]

    return log_mel_spectrogram


if quantization_type == "none":
    converter.optimizations = []

elif quantization_type == "dynamic":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

elif quantization_type == "float16":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

elif quantization_type == "int8":
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    val_files, val_labels, _ = load_data()

    # You need a representative dataset for full integer quantization
    def representative_dataset():
        for file, label in zip(val_files[:100], val_labels[:100]):
            try:
                mel_db = preprocess_audio(file)
                mel_db = np.expand_dims(mel_db, axis=-1)
                mel_db = mel_db.astype(np.float32)
                yield [mel_db]
            except Exception as e:
                print(f"Error in generator for {file}: {e}")
                exit(1)

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8  # or tf.int8
    converter.inference_output_type = tf.uint8  # or tf.int8

# ==== Convert ====
tflite_model = converter.convert()

# ==== Save the TFLite model ====
with open(f"{tf_lite_model_name}.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved model!")
