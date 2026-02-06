import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
import os

print("Fixing MLP model...")

# ---- FIX MLP ----
mlp = Sequential([
    Input(shape=(784,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

mlp.load_weights("../saved_models/mlp_model.keras", by_name=True, skip_mismatch=True)
mlp.save("../saved_models/mlp_model_fixed.keras")

print("MLP fixed")

# ---- FIX CNN ----
print("Fixing CNN model...")

cnn = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

cnn.load_weights("../saved_models/cnn_model.keras", by_name=True, skip_mismatch=True)
cnn.save("../saved_models/cnn_model_fixed.keras")

print("CNN fixed")
print("ALL MODELS CONVERTED SUCCESSFULLY")
