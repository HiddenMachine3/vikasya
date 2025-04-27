import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_DEVICE_PARTITIONING"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device

import warnings

# Suppress specific UserWarnings from librosa
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"  # Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_DEVICE_PARTITIONING"] = (
    "0"  # Suppress TensorFlow device partitioning
)

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

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

    train_files, test_files, train_labels, test_labels = train_test_split(
        files, labels, test_size=0.2, random_state=42
    )
    return train_files, test_files, train_labels, test_labels


def preprocess_audio(filepath, sample_rate=16000):
    y, sr = sf.read(filepath)

    if sr != sample_rate:
        y = librosa.resample(y=y, orig_sr=sr, target_sr=sample_rate)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    y = y / (np.max(np.abs(y)) + 1e-10)

    # 1. Log-Mel Spectrogram
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

    # 2. MFCC + Delta + Delta Delta
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc])
    mfcc_features = np.resize(mfcc_features, (39, 32))

    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    chroma = np.resize(chroma, (12, 32))

    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)
    contrast = np.resize(contrast, (7, 32))

    # 5. Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sample_rate)
    tonnetz = np.resize(tonnetz, (6, 32))

    # 6. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = np.resize(zcr, (1, 32))

    # 7. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
    rolloff = np.resize(rolloff, (1, 32))

    # 8. Fundamental Frequency (YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sample_rate
    )
    f0 = np.nan_to_num(f0)
    f0 = f0.reshape(1, -1)
    f0 = np.resize(f0, (1, 32))

    # 9. Voice Activity Detection mask
    # Simple energy-based VAD
    frame_length = 512
    hop_length = 256
    rmse = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    threshold = 0.1 * np.max(rmse)
    vad_mask = (rmse > threshold).astype(np.float32)
    vad_mask = np.resize(vad_mask, (1, 32))

    # Combine all features
    combined_features = np.vstack(
        [
            log_mel_spectrogram,  # 64x32
            mfcc_features,  # 39x32
            chroma,  # 12x32
            contrast,  # 7x32
            tonnetz,  # 6x32
            zcr,  # 1x32
            rolloff,  # 1x32
            f0,  # 1x32
            vad_mask,  # 1x32
        ]
    )

    # Normalize all features
    for i in range(combined_features.shape[0]):
        if np.std(combined_features[i]) > 0:
            combined_features[i] = (
                combined_features[i] - np.mean(combined_features[i])
            ) / np.std(combined_features[i])

    return combined_features


def audio_dataset_generator(files, labels):
    for file, label in zip(files, labels):
        try:
            features = preprocess_audio(file)
            features = np.expand_dims(features, axis=-1)
            yield features, label
        except Exception as e:
            print(f"Error in generator for {file}: {e}")
            continue


def build_model(
    input_shape=(132, 32, 1),
):  # Updated input shape: 64+39+12+7+6+1+1+1+1 = 132
    with tf.device(DEVICE):
        # Convert the input shape to have 3 channels (grayscale to RGB)
        input_layer = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(
            input_layer
        )  # Convert to 3 channels

        # Load ResNet50 pre-trained model without the top layers (include_top=False)
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(132, 32, 3),  # Now input_shape is 3 channels
        )

        # Freeze the layers of ResNet50
        base_model.trainable = False

        # Create a custom model on top
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(2, activation="softmax")(x)

        # Define the full model
        model = tf.keras.Model(inputs=input_layer, outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def main():
    choice = input("Train Also? Y/N: ")
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)

    print(f"Using device: {DEVICE}")
    print(f"Looking for data in: {AUDIO_DIR}")

    if not os.path.exists(REAL_DIR) or not os.path.exists(FAKE_DIR):
        print(
            "Error: Data directories not found. Please ensure the following paths exist:"
        )
        print(f"  - {REAL_DIR}")
        print(f"  - {FAKE_DIR}")
        return

    train_files, test_files, train_labels, test_labels = load_data()
    print(
        f"Loaded {len(train_files)} training files and {len(test_files)} testing files"
    )

    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: audio_dataset_generator(train_files, train_labels),
            output_signature=(
                tf.TensorSpec(shape=(132, 32, 1), dtype=tf.float32),  # Updated shape
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
        .repeat()
    )

    test_dataset = (
        tf.data.Dataset.from_generator(
            lambda: audio_dataset_generator(test_files, test_labels),
            output_signature=(
                tf.TensorSpec(shape=(132, 32, 1), dtype=tf.float32),  # Updated shape
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )

    model = build_model()
    model.summary()

    if choice.lower() == "y":
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.keras", save_best_only=True, monitor="val_accuracy"
            ),
        ]

        print("Training the model...")

        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset,
            callbacks=callbacks,
            steps_per_epoch=len(train_files) // BATCH_SIZE,
            verbose=1,
            shuffle=True,
            initial_epoch=0,
            validation_steps=len(test_files) // BATCH_SIZE,
            validation_batch_size=BATCH_SIZE,
            validation_freq=1,
        )

    # model.load_weights("final_model.keras")
    test_preds = []
    true_labels = []

    print("Evaluating on test set...")

    for batch_x, batch_y in test_dataset:
        batch_preds = model.predict(batch_x, verbose=0)
        batch_preds_labels = np.argmax(batch_preds, axis=1)
        test_preds.extend(batch_preds_labels)
        true_labels.extend(batch_y.numpy())

    acc = accuracy_score(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds)
    print(f"Test Accuracy: {acc:.4f} | Test F1 Score: {f1:.4f}")

    if choice.lower() == "y":
        model.save("final_model.keras")
        print("Model saved as 'final_model.keras'")
        return


if __name__ == "__main__":
    main()
