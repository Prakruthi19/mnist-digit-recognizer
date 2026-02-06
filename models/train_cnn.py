"""
Train a Convolutional Neural Network (CNN) on MNIST Dataset
CNN understands spatial patterns and relationships in images
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
import os

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

print("=" * 60)
print("TRAINING CNN (CONVOLUTIONAL NEURAL NETWORK) ON MNIST")
print("=" * 60)
print()

# ============================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================
print("📊 STEP 1: Loading MNIST Dataset...")
print("-" * 60)

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✅ Training data shape: {X_train.shape}")
print(f"✅ Training labels shape: {y_train.shape}")
print(f"✅ Test data shape: {X_test.shape}")
print(f"✅ Test labels shape: {y_test.shape}")
print()

# ============================================================
# STEP 2: PREPROCESS DATA (DIFFERENT FROM MLP!)
# ============================================================
print("🔧 STEP 2: Preprocessing Data for CNN...")
print("-" * 60)

# CNN keeps the 2D structure! No flattening!
# Add channel dimension (1 for grayscale)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

print(f"✅ Reshaped training data: {X_train_cnn.shape}")
print(f"   Format: (samples, height, width, channels)")
print(f"   • Samples: {X_train_cnn.shape[0]}")
print(f"   • Height: {X_train_cnn.shape[1]} pixels")
print(f"   • Width: {X_train_cnn.shape[2]} pixels")
print(f"   • Channels: {X_train_cnn.shape[3]} (1 = grayscale)")
print()

print("🔍 KEY DIFFERENCE FROM MLP:")
print("   MLP:  (60000, 784)      - Flattened, loses spatial info")
print("   CNN:  (60000, 28, 28, 1) - Keeps 2D structure!")
print()

# Normalize pixel values to [0, 1]
X_train_cnn = X_train_cnn.astype('float32') / 255.0
X_test_cnn = X_test_cnn.astype('float32') / 255.0

print(f"✅ Normalized pixel range: {X_train_cnn.min():.2f} to {X_train_cnn.max():.2f}")

# One-hot encode labels
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

print(f"✅ Encoded labels shape: {y_train_encoded.shape}")
print()

# ============================================================
# STEP 3: BUILD CNN MODEL
# ============================================================
print("🧠 STEP 3: Building CNN Architecture...")
print("-" * 60)

model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                  input_shape=(28, 28, 1), name='conv1'),
    layers.BatchNormalization(name='bn1'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
    
    # Second Convolutional Block
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
    layers.BatchNormalization(name='bn2'),
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
    
    # Third Convolutional Block
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu', name='conv3'),
    layers.BatchNormalization(name='bn3'),
    
    # Flatten and Dense Layers
    layers.Flatten(name='flatten'),
    layers.Dropout(0.3, name='dropout1'),
    layers.Dense(256, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout2'),
    
    # Output Layer
    layers.Dense(10, activation='softmax', name='output')
], name='CNN_MNIST')

print("✅ CNN Model Architecture:")
print()
model.summary()
print()

# Total parameters
total_params = model.count_params()
print(f"📊 Total trainable parameters: {total_params:,}")
print()

print("🔍 LAYER EXPLANATION:")
print("-" * 60)
print("Conv2D:    Detects patterns (edges, curves, shapes)")
print("           • 32, 64, 128 = number of filters/patterns")
print("           • (3,3) = size of filter window")
print()
print("MaxPooling: Reduces size, keeps important features")
print("           • (2,2) = reduces by half")
print()
print("BatchNorm: Normalizes activations (faster training)")
print()
print("Flatten:   Converts 2D → 1D for dense layers")
print()
print("Dropout:   Randomly drops neurons (prevents overfitting)")
print()

# ============================================================
# STEP 4: COMPILE MODEL
# ============================================================
print("⚙️  STEP 4: Compiling Model...")
print("-" * 60)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Optimizer: Adam")
print("✅ Loss function: Categorical Crossentropy")
print("✅ Metrics: Accuracy")
print()

# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================
print("🚀 STEP 5: Training CNN Model...")
print("-" * 60)
print("⏳ This will take 3-7 minutes depending on your hardware...")
print("   (CNNs train slower than MLPs but achieve better accuracy!)")
print()

# Define callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        'saved_models/best_cnn_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
history = model.fit(
    X_train_cnn,
    y_train_encoded,
    batch_size=128,
    epochs=20,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

print()
print("✅ Training completed!")
print()

# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================
print("📈 STEP 6: Evaluating CNN Model...")
print("-" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_encoded, verbose=0)

print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print()

# Make predictions
predictions = model.predict(X_test_cnn, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test

# Calculate metrics
correct_predictions = np.sum(predicted_classes == true_classes)
incorrect_predictions = len(true_classes) - correct_predictions

print(f"✅ Correct predictions: {correct_predictions}/{len(true_classes)}")
print(f"❌ Incorrect predictions: {incorrect_predictions}/{len(true_classes)}")
print()

# ============================================================
# STEP 7: COMPARE WITH MLP
# ============================================================
print("🆚 STEP 7: MLP vs CNN Comparison...")
print("-" * 60)

# Try to load MLP results if they exist
try:
    mlp_model = keras.models.load_model('saved_models/mlp_model.keras')
    X_test_flat = X_test.reshape(-1, 784).astype('float32') / 255.0
    mlp_loss, mlp_accuracy = mlp_model.evaluate(X_test_flat, y_test_encoded, verbose=0)
    
    print("📊 COMPARISON:")
    print(f"   MLP Accuracy:  {mlp_accuracy * 100:.2f}%")
    print(f"   CNN Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"   Improvement:   +{(test_accuracy - mlp_accuracy) * 100:.2f}%")
    print()
    
    if test_accuracy > mlp_accuracy:
        print("🎉 CNN wins! It understands spatial patterns better!")
    
except:
    print("⚠️  MLP model not found. Train MLP first to compare.")
    print(f"   CNN Accuracy: {test_accuracy * 100:.2f}%")

print()

# ============================================================
# STEP 8: VISUALIZE TRAINING HISTORY
# ============================================================
print("📊 STEP 8: Visualizing Training History...")
print("-" * 60)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
ax1.set_title('CNN Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
ax2.set_title('CNN Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_training_history.png', dpi=150, bbox_inches='tight')
print("✅ Saved training history plot to 'cnn_training_history.png'")
plt.close()
print()

# ============================================================
# STEP 9: VISUALIZE PREDICTIONS
# ============================================================
print("🔍 STEP 9: Visualizing CNN Predictions...")
print("-" * 60)

# Show correct and incorrect predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('CNN Predictions (Green=Correct, Red=Wrong)', fontsize=16)

# Show 5 correct predictions
correct_indices = np.where(predicted_classes == true_classes)[0][:5]
for i, ax in enumerate(axes[0]):
    idx = correct_indices[i]
    confidence = np.max(predictions[idx]) * 100
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f'True: {true_classes[idx]}\nPred: {predicted_classes[idx]}\n({confidence:.1f}%)', 
                 color='green', fontweight='bold')
    ax.axis('off')

# Show 5 incorrect predictions
incorrect_indices = np.where(predicted_classes != true_classes)[0][:5]
for i, ax in enumerate(axes[1]):
    if i < len(incorrect_indices):
        idx = incorrect_indices[i]
        confidence = np.max(predictions[idx]) * 100
        ax.imshow(X_test[idx], cmap='gray')
        ax.set_title(f'True: {true_classes[idx]}\nPred: {predicted_classes[idx]}\n({confidence:.1f}%)', 
                     color='red', fontweight='bold')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('cnn_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Saved predictions visualization to 'cnn_predictions.png'")
plt.close()
print()

# ============================================================
# STEP 10: VISUALIZE FILTERS (WHAT CNN LEARNED)
# ============================================================
print("🔬 STEP 10: Visualizing What CNN Learned...")
print("-" * 60)

# Get first convolutional layer
conv_layer = model.get_layer('conv1')
filters, biases = conv_layer.get_weights()

print(f"First layer filters shape: {filters.shape}")
print(f"   • Filter size: {filters.shape[0]}×{filters.shape[1]}")
print(f"   • Input channels: {filters.shape[2]}")
print(f"   • Number of filters: {filters.shape[3]}")
print()

# Normalize filters for visualization
f_min, f_max = filters.min(), filters.max()
filters_normalized = (filters - f_min) / (f_max - f_min)

# Plot first 16 filters
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('CNN First Layer Filters (Patterns it Detects)', fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < filters_normalized.shape[3]:
        # Get the filter
        filter_img = filters_normalized[:, :, 0, i]
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig('cnn_filters.png', dpi=150, bbox_inches='tight')
print("✅ Saved filter visualization to 'cnn_filters.png'")
print("   These are the patterns CNN uses to detect edges, curves, etc.")
plt.close()
print()

# ============================================================
# STEP 11: SAVE MODEL
# ============================================================
print("💾 STEP 11: Saving CNN Model...")
print("-" * 60)

# Save the model
model.save('saved_models/cnn_model.keras')
print("✅ Model saved to 'saved_models/cnn_model.keras'")
print()

# Save model architecture as JSON
model_json = model.to_json()
with open('saved_models/cnn_model.json', 'w') as json_file:
    json_file.write(model_json)
print("✅ Model architecture saved to 'saved_models/cnn_model.json'")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("🎉 CNN TRAINING COMPLETE!")
print("=" * 60)
print()
print("📊 SUMMARY:")
print(f"   • Model: Convolutional Neural Network (CNN)")
print(f"   • Architecture: Conv(32) → Conv(64) → Conv(128) → Dense(256) → 10")
print(f"   • Total Parameters: {total_params:,}")
print(f"   • Training Samples: {len(X_train):,}")
print(f"   • Test Samples: {len(X_test):,}")
print(f"   • Final Test Accuracy: {test_accuracy * 100:.2f}%")
print()
print("📁 GENERATED FILES:")
print("   • cnn_training_history.png")
print("   • cnn_predictions.png")
print("   • cnn_filters.png (what CNN learned!)")
print("   • saved_models/cnn_model.keras")
print("   • saved_models/cnn_model.json")
print()
print("🎯 NEXT STEPS:")
print("   1. Compare cnn_filters.png to see what patterns CNN detects")
print("   2. Notice CNN accuracy is higher than MLP!")
print("   3. Ready to build the web application!")
print()
print("=" * 60)