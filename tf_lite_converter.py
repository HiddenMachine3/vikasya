import tensorflow as tf

model_name = "models/best_model_image_analysis_84_val_acc"  # or .keras or SavedModel

tf_lite_model_name = "models/best_model_image_analysis_84_val_acc"  # or .keras or SavedModel

# Load your Keras model
model = tf.keras.models.load_model(f"{model_name}.keras")  # or .keras or SavedModel

# Set up the TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Make sure optimizations are off
converter.optimizations = []

# Convert the model
tflite_model = converter.convert()

# Save the TFLite model
with open(f"{tf_lite_model_name}.tflite", "wb") as f:
    f.write(tflite_model)
print("Saved model!")