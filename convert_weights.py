import os
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

print("Loading DenseNet121...")
model = keras.models.load_model('models/densenet121.keras')
model.summary()
print("Saving weights to .h5...")
model.save_weights('models/densenet121.weights.h5')
print("Done!")
