import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model(
    "best_model_12.46_89_test.keras"
)  # or .keras or SavedModel

# Set up the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Make sure optimizations are off
converter.optimizations = []

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open("best_model_12.46_89_test.tflite", "wb") as f:
    f.write(tflite_model)
