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
print("TRAINING MLP (MULTI-LAYER PERCEPTRON) ON MNIST")
print("=" * 60)
print()

# ============================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================
print("📊 STEP 1: Loading MNIST Dataset...")
print("-" * 60)

# Load MNIST data (60,000 training images, 10,000 test images)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"✅ Training data shape: {X_train.shape}")  # (60000, 28, 28)
print(f"✅ Training labels shape: {y_train.shape}")  # (60000,)
print(f"✅ Test data shape: {X_test.shape}")  # (10000, 28, 28)
print(f"✅ Test labels shape: {y_test.shape}")  # (10000,)
print(f"✅ Pixel value range: {X_train.min()} to {X_train.max()}")
print()

# Visualize sample images
print("📷 Sample images from dataset:")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample MNIST Images', fontsize=16)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(f'Label: {y_train[i]}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_mnist_images.png', dpi=150, bbox_inches='tight')
print("✅ Saved sample images to 'sample_mnist_images.png'")
plt.close()
print()

# ============================================================
# STEP 2: PREPROCESS DATA
# ============================================================
print("🔧 STEP 2: Preprocessing Data...")
print("-" * 60)

# MLP treats images as flat vectors (not 2D images!)
# Flatten 28x28 images to 784-dimensional vectors
X_train_flat = X_train.reshape(-1, 28 * 28)  # (60000, 784)
X_test_flat = X_test.reshape(-1, 28 * 28)    # (10000, 784)

print(f"✅ Flattened training data: {X_train_flat.shape}")
print(f"✅ Flattened test data: {X_test_flat.shape}")

# Normalize pixel values from [0, 255] to [0, 1]
X_train_flat = X_train_flat.astype('float32') / 255.0
X_test_flat = X_test_flat.astype('float32') / 255.0

print(f"✅ Normalized pixel range: {X_train_flat.min():.2f} to {X_train_flat.max():.2f}")

# One-hot encode labels
# Example: 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train_encoded = to_categorical(y_train, 10)
y_test_encoded = to_categorical(y_test, 10)

print(f"✅ Encoded labels shape: {y_train_encoded.shape}")
print(f"   Example: Label {y_train[0]} → {y_train_encoded[0]}")
print()

# ============================================================
# STEP 3: BUILD MLP MODEL
# ============================================================
print("🧠 STEP 3: Building MLP Architecture...")
print("-" * 60)

model = models.Sequential([
    # Input layer: 784 neurons (28x28 pixels flattened)
    layers.Dense(512, activation='relu', input_shape=(784,), name='dense_1'),
    layers.Dropout(0.2, name='dropout_1'),  # Prevent overfitting
    
    # Hidden layer
    layers.Dense(256, activation='relu', name='dense_2'),
    layers.Dropout(0.2, name='dropout_2'),
    
    # Hidden layer
    layers.Dense(128, activation='relu', name='dense_3'),
    layers.Dropout(0.2, name='dropout_3'),
    
    # Output layer: 10 neurons (digits 0-9)
    layers.Dense(10, activation='softmax', name='output')
], name='MLP_MNIST')

print("✅ MLP Model Architecture:")
print()
model.summary()
print()

# Total parameters
total_params = model.count_params()
print(f"📊 Total trainable parameters: {total_params:,}")
print()

# ============================================================
# STEP 4: COMPILE MODEL
# ============================================================
print("⚙️  STEP 4: Compiling Model...")
print("-" * 60)

model.compile(
    optimizer='adam',  # Adaptive learning rate optimizer
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']  # Track accuracy during training
)

print("✅ Optimizer: Adam")
print("✅ Loss function: Categorical Crossentropy")
print("✅ Metrics: Accuracy")
print()

# ============================================================
# STEP 5: TRAIN MODEL
# ============================================================
print("🚀 STEP 5: Training Model...")
print("-" * 60)
print("⏳ This will take 2-5 minutes depending on your hardware...")
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
    )
]

# Train the model
history = model.fit(
    X_train_flat, 
    y_train_encoded,
    batch_size=128,
    epochs=20,
    validation_split=0.1,  # Use 10% of training data for validation
    callbacks=callbacks,
    verbose=1
)

print()
print("✅ Training completed!")
print()

# ============================================================
# STEP 6: EVALUATE MODEL
# ============================================================
print("📈 STEP 6: Evaluating Model...")
print("-" * 60)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test_encoded, verbose=0)

print(f"✅ Test Loss: {test_loss:.4f}")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print()

# Make predictions on test set
predictions = model.predict(X_test_flat, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test

# Calculate metrics
correct_predictions = np.sum(predicted_classes == true_classes)
incorrect_predictions = len(true_classes) - correct_predictions

print(f"✅ Correct predictions: {correct_predictions}/{len(true_classes)}")
print(f"❌ Incorrect predictions: {incorrect_predictions}/{len(true_classes)}")
print()

# ============================================================
# STEP 7: VISUALIZE RESULTS
# ============================================================
print("📊 STEP 7: Visualizing Training History...")
print("-" * 60)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('MLP Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('MLP Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_training_history.png', dpi=150, bbox_inches='tight')
print("✅ Saved training history plot to 'mlp_training_history.png'")
plt.close()
print()

# ============================================================
# STEP 8: VISUALIZE PREDICTIONS
# ============================================================
print("🔍 STEP 8: Visualizing Sample Predictions...")
print("-" * 60)

# Show some correct and incorrect predictions
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('MLP Predictions (Green=Correct, Red=Wrong)', fontsize=16)

# Show 5 correct predictions
correct_indices = np.where(predicted_classes == true_classes)[0][:5]
for i, ax in enumerate(axes[0]):
    idx = correct_indices[i]
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f'True: {true_classes[idx]}\nPred: {predicted_classes[idx]}', 
                 color='green', fontweight='bold')
    ax.axis('off')

# Show 5 incorrect predictions
incorrect_indices = np.where(predicted_classes != true_classes)[0][:5]
for i, ax in enumerate(axes[1]):
    idx = incorrect_indices[i]
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f'True: {true_classes[idx]}\nPred: {predicted_classes[idx]}', 
                 color='red', fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mlp_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Saved predictions visualization to 'mlp_predictions.png'")
plt.close()
print()

# ============================================================
# STEP 9: SAVE MODEL
# ============================================================
print("💾 STEP 9: Saving Model...")
print("-" * 60)

# Save the model
model.save('saved_models/mlp_model.keras')
print("✅ Model saved to 'saved_models/mlp_model.keras'")
print()

# Save model architecture as JSON
model_json = model.to_json()
with open('saved_models/mlp_model.json', 'w') as json_file:
    json_file.write(model_json)
print("✅ Model architecture saved to 'saved_models/mlp_model.json'")
print()

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
print()
print("📊 SUMMARY:")
print(f"   • Model: Multi-Layer Perceptron (MLP)")
print(f"   • Architecture: 784 → 512 → 256 → 128 → 10")
print(f"   • Total Parameters: {total_params:,}")
print(f"   • Training Samples: {len(X_train):,}")
print(f"   • Test Samples: {len(X_test):,}")
print(f"   • Final Test Accuracy: {test_accuracy * 100:.2f}%")
print()
print("📁 GENERATED FILES:")
print("   • sample_mnist_images.png")
print("   • mlp_training_history.png")
print("   • mlp_predictions.png")
print("   • saved_models/mlp_model.keras")
print("   • saved_models/mlp_model.json")
print()
print("🎯 NEXT STEPS:")
print("   1. Check the generated images to see model performance")
print("   2. Run 'models/train_cnn.py' to train CNN and compare")
print("   3. Use the saved model in the web application")
print()
print("=" * 60)