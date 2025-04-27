import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Constants
BATCH_SIZE = 16
AUDIO_DIR = os.path.join(os.getcwd(), "AUDIO_DATASET")
REAL_DIR = os.path.join(AUDIO_DIR, "real_wav")
FAKE_DIR = os.path.join(AUDIO_DIR, "fake_wav")
TFLITE_MODEL_PATH = "best_model_12.46_89_test.tflite"  # Path to your TFLite model


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

    return test_files, test_labels


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

    n_frames = 32
    if log_mel_spectrogram.shape[1] < n_frames:
        log_mel_spectrogram = np.pad(
            log_mel_spectrogram,
            ((0, 0), (0, n_frames - log_mel_spectrogram.shape[1])),
            mode="constant",
        )
    else:
        log_mel_spectrogram = log_mel_spectrogram[:, :n_frames]

    return log_mel_spectrogram


def run_tflite_inference(tflite_model_path, test_files, test_labels):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]

    test_preds = []
    true_labels = []

    for file, label in zip(test_files, test_labels):
        try:
            feature = preprocess_audio(file)
            feature = np.expand_dims(feature, axis=-1)  # Add channel dimension
            feature = np.expand_dims(feature, axis=0)  # Add batch dimension
            feature = feature.astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], feature)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])
            pred_label = np.argmax(output_data, axis=1)[0]

            test_preds.append(pred_label)
            true_labels.append(label)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    acc = accuracy_score(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds)

    print(f"\n[TFLITE] Test Accuracy: {acc:.4f} | Test F1 Score: {f1:.4f}")


def main():
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"TFLite model not found at {TFLITE_MODEL_PATH}")
        return

    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print("Data folders not found.")
        return

    test_files, test_labels = load_data()
    print(f"Loaded {len(test_files)} test files.")

    run_tflite_inference(TFLITE_MODEL_PATH, test_files, test_labels)


if __name__ == "__main__":
    main()
