"""
Image preprocessing and prediction functions
"""

import numpy as np
from PIL import Image, ImageOps
import cv2

def preprocess_image(image, model_type='cnn', debug=True):
    """
    Preprocess PIL image for model prediction
    
    Args:
        image: PIL Image object
        model_type: 'mlp' or 'cnn'
        debug: Print debug information
    
    Returns:
        Preprocessed numpy array ready for prediction
    """
    
    if debug:
        print(f"\n{'='*50}")
        print(f"🔍 PREPROCESSING DEBUG ({model_type.upper()})")
        print(f"{'='*50}")
        print(f"Original image size: {image.size}")
        print(f"Original image mode: {image.mode}")
    
    # Convert to grayscale
    image = image.convert('L')
    
    if debug:
        img_before_resize = np.array(image)
        print(f"After grayscale - mean pixel value: {img_before_resize.mean():.2f}")
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Check if we need to invert
    img_array_check = np.array(image)
    mean_value = img_array_check.mean()
    
    if debug:
        print(f"After resize - mean pixel value: {mean_value:.2f}")
        print(f"Pixel range: [{img_array_check.min()}, {img_array_check.max()}]")
    
    # Invert colors (MNIST has white digits on black background)
    # Only invert if background is lighter than foreground
    image = ImageOps.invert(image)
    
    if debug:
        img_after_invert = np.array(image)
        print(f"After invert - mean pixel value: {img_after_invert.mean():.2f}")
        print(f"After invert - pixel range: [{img_after_invert.min()}, {img_after_invert.max()}]")
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    if debug:
        print(f"After normalization - range: [{img_array.min():.4f}, {img_array.max():.4f}]")
        print(f"After normalization - mean: {img_array.mean():.4f}")
    
    # Reshape based on model type
    if model_type == 'mlp':
        # MLP expects flattened input (1, 784)
        img_array = img_array.reshape(1, 784)
    else:
        # CNN expects 2D input with channel (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
    
    if debug:
        print(f"Final shape: {img_array.shape}")
        print(f"{'='*50}\n")
    
    # Save debug image
    if debug:
        import matplotlib.pyplot as plt
        plt.imsave('debug_preprocessed.png', img_array.reshape(28, 28), cmap='gray')
        print("💾 Saved debug_preprocessed.png")
    
    return img_array

def predict_digit(model, preprocessed_image, debug=True):
    """
    Make prediction using the model
    
    Args:
        model: Trained Keras model
        preprocessed_image: Preprocessed image array
        debug: Print debug information
    
    Returns:
        Dictionary with prediction results
    """
    
    # Get prediction probabilities
    predictions = model.predict(preprocessed_image, verbose=0)[0]
    
    if debug:
        print(f"\n{'='*50}")
        print(f"🎯 PREDICTION DEBUG")
        print(f"{'='*50}")
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Sum of probabilities: {predictions.sum():.4f}")
    
    # Get predicted digit (index of max probability)
    predicted_digit = int(np.argmax(predictions))
    
    # Get confidence (max probability)
    confidence = float(predictions[predicted_digit])
    
    if debug:
        print(f"\nTop 3 predictions:")
        sorted_indices = np.argsort(predictions)[::-1][:3]
        for i, idx in enumerate(sorted_indices, 1):
            print(f"  {i}. Digit {idx}: {predictions[idx]*100:.2f}%")
        print(f"{'='*50}\n")
    
    # Get all class probabilities
    all_probabilities = {
        str(i): float(predictions[i]) for i in range(10)
    }
    
    # Sort by confidence (descending)
    sorted_probs = dict(sorted(
        all_probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    ))
    
    return {
        'digit': predicted_digit,
        'confidence': confidence,
        'all_probabilities': all_probabilities,
        'top_3': dict(list(sorted_probs.items())[:3])
    }