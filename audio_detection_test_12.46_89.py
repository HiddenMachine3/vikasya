import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow GPU memory growth
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"  # Suppress TensorFlow verbose logging
os.environ["TF_CPP_MIN_DEVICE_PARTITIONING"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify GPU device

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

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

    return train_files, val_files, test_files, train_labels, val_labels, test_labels


def run_tflite_inference(tflite_model_path, test_dataset):
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_preds = []
    true_labels = []

    for batch_x, batch_y in test_dataset:
        batch_x_numpy = batch_x.numpy()

        for i in range(batch_x_numpy.shape[0]):
            input_data = np.expand_dims(batch_x_numpy[i], axis=0)  # add batch dimension

            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            pred_label = np.argmax(output_data, axis=1)[0]
            test_preds.append(pred_label)
            true_labels.append(batch_y.numpy()[i])

    acc = accuracy_score(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds)
    print(f"[TFLITE] Test Accuracy: {acc:.4f} | Test F1 Score: {f1:.4f}")


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


def audio_dataset_generator(files, labels):
    for file, label in zip(files, labels):
        try:
            mel_db = preprocess_audio(file)
            mel_db = np.expand_dims(mel_db, axis=-1)
            mel_db = mel_db.astype(np.float32)
            yield mel_db, label
        except Exception as e:
            print(f"Error in generator for {file}: {e}")

            continue


def build_model(input_shape=(64, 32, 1)):
    with tf.device(DEVICE):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
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

    train_files, val_files, test_files, train_labels, val_labels, test_labels = (
        load_data()
    )
    print(
        f"Loaded {len(train_files)} training files and {len(test_files)} testing files"
    )

    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: audio_dataset_generator(train_files, train_labels),
            output_signature=(
                tf.TensorSpec(shape=(64, 32, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    val_dataset = (
        tf.data.Dataset.from_generator(
            lambda: audio_dataset_generator(val_files, val_labels),
            output_signature=(
                tf.TensorSpec(shape=(64, 32, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_generator(
            lambda: audio_dataset_generator(test_files, test_labels),
            output_signature=(
                tf.TensorSpec(shape=(64, 32, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
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

        model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=callbacks,
            steps_per_epoch=len(train_files) // BATCH_SIZE,
            verbose=1,
            shuffle=True,
            initial_epoch=0,
            validation_steps=len(val_files) // BATCH_SIZE,
            validation_batch_size=BATCH_SIZE,
            validation_freq=1,
        )

    model.load_weights("best_model_12.46_89_test.keras")
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

    # Normal Keras model evaluation already done above
    print("\n--- Now evaluating TFLite model ---")
    run_tflite_inference("best_model_12.46_89_test.tflite", test_dataset)


if __name__ == "__main__":
    main()
