"""
Image preprocessing and prediction functions
"""

import numpy as np
from PIL import Image, ImageOps
import cv2

def preprocess_image(image, model_type='cnn'):
    """
    Preprocess PIL image for model prediction
    
    Args:
        image: PIL Image object
        model_type: 'mlp' or 'cnn'
    
    Returns:
        Preprocessed numpy array ready for prediction
    """
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Invert colors (MNIST has white digits on black background)
    image = ImageOps.invert(image)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Reshape based on model type
    if model_type == 'mlp':
        # MLP expects flattened input (1, 784)
        img_array = img_array.reshape(1, 784)
    else:
        # CNN expects 2D input with channel (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def predict_digit(model, preprocessed_image):
    """
    Make prediction using the model
    
    Args:
        model: Trained Keras model
        preprocessed_image: Preprocessed image array
    
    Returns:
        Dictionary with prediction results
    """
    
    # Get prediction probabilities
    predictions = model.predict(preprocessed_image, verbose=0)[0]
    
    # Get predicted digit (index of max probability)
    predicted_digit = int(np.argmax(predictions))
    
    # Get confidence (max probability)
    confidence = float(predictions[predicted_digit])
    
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