import os
import subprocess

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

from backend.analysis.base_analyzer import BaseAnalyzer


class AudioDeepfakeAnalyzer(BaseAnalyzer):
    """
    Analyzes audio files to detect potential deepfake audio
    using a TFLite model trained for this purpose.
    """

    def __init__(self):
        """Initialize the analyzer with model path and parameters."""
        self.tflite_model_path = "best_model_12.46_89_test.tflite"
        self.sample_rate = 16000
        self.n_mels = 64
        self.n_frames = 32

    def convert_to_wav(self, file_path):
        """Converts .flac or .mp3 file to .wav using ffmpeg."""
        if not file_path.endswith(".wav"):
            output_path = os.path.splitext(file_path)[0] + ".wav"
            subprocess.run(["ffmpeg", "-i", file_path, output_path], check=True)
            return output_path
        return file_path

    def preprocess_audio(self, filepath):
        """
        Preprocess audio file for model input.

        Args:
            filepath: Path to the audio file.

        Returns:
            Preprocessed audio features as a mel spectrogram.
        """

        filepath = self.convert_to_wav(filepath)

        y, sr = sf.read(filepath)

        if sr != self.sample_rate:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=self.sample_rate)

        if len(y.shape) > 1:
            y = np.mean(y, axis=1)

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_mels=self.n_mels
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spectrogram = (
            log_mel_spectrogram - np.mean(log_mel_spectrogram)
        ) / np.std(log_mel_spectrogram)

        if log_mel_spectrogram.shape[1] < self.n_frames:
            log_mel_spectrogram = np.pad(
                log_mel_spectrogram,
                ((0, 0), (0, self.n_frames - log_mel_spectrogram.shape[1])),
                mode="constant",
            )
        else:
            log_mel_spectrogram = log_mel_spectrogram[:, : self.n_frames]

        return log_mel_spectrogram

    def analyze(self, file_path: str) -> dict:
        """
        Analyzes an audio file to determine if it's likely a deepfake.

        Args:
            file_path: Path to the audio file.

        Returns:
            A dictionary with analysis results.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        if not os.path.exists(self.tflite_model_path):
            return {"error": f"TFLite model not found at {self.tflite_model_path}"}

        try:
            # Load and set up the TFLite model
            interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Process the audio file
            feature = self.preprocess_audio(file_path)
            feature = np.expand_dims(feature, axis=-1)  # Add channel dimension
            feature = np.expand_dims(feature, axis=0)  # Add batch dimension
            feature = feature.astype(np.float32)

            # Run inference
            interpreter.set_tensor(input_details[0]["index"], feature)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])

            # Get prediction (0 = real, 1 = fake)
            pred_label = np.argmax(output_data, axis=1)[0]
            confidence = float(output_data[0][pred_label])

            result = {
                "analyzer": "Audio Deepfake Detector",
                "file": file_path,
                "prediction": "FAKE" if pred_label == 1 else "REAL",
                "confidence": round(confidence, 4),
                "raw_scores": {
                    "real_score": float(output_data[0][0]),
                    "fake_score": float(output_data[0][1]),
                },
            }

            return result

        except Exception as e:
            return {"error": f"Audio deepfake analysis failed: {str(e)}"}
