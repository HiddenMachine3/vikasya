import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
VIDEO_DIR = os.path.join(os.getcwd(), "DFD_sampled")
REAL_DIR = os.path.join(VIDEO_DIR, "real")
FAKE_DIR = os.path.join(VIDEO_DIR, "fake")
DEVICE = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"


def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

    cap.release()
    return np.array(frames)


def load_data(video_dir, label, num_videos=50, frames_per_video=10):
    video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
    data = []
    labels = []

    for video_path in tqdm(video_paths[:num_videos], desc=f"Processing {label} videos"):
        frames = extract_frames(video_path, frames_per_video)
        data.extend(frames)
        labels.extend([label] * len(frames))

    return np.array(data), np.array(labels)


def create_finetuned_efficientnetb7_model(input_shape=(224, 224, 3)):
    base_model = EfficientNetB7(
        weights="imagenet", include_top=False, input_shape=input_shape
    )

    # Unfreeze the last few layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential(
        [
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.4),  # Adjusted dropout
            Dense(512, activation="relu"),  # Increased layer size
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


X_fake, y_fake = load_data(FAKE_DIR, 1)
X_real, y_real = load_data(REAL_DIR, 0)

# Combine and shuffle data
X = np.concatenate((X_fake, X_real), axis=0)
y = np.concatenate((y_fake, y_real), axis=0)

# Shuffle the dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

with tf.device(DEVICE):
    model = create_finetuned_efficientnetb7_model()
    model.load_weights("deepfake_detector.keras")

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=16, shuffle=False)

    y_pred = model.predict(val_generator)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()

# Classification report
print(classification_report(y_val, y_pred_classes, target_names=["Real", "Fake"]))
