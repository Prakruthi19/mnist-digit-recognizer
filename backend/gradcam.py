"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualizes what the CNN focuses on when making predictions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import base64
from io import BytesIO
from PIL import Image

def generate_gradcam(model, image, predicted_class, layer_name='conv3'):
    """
    Generate Grad-CAM heatmap for CNN predictions
    
    Args:
        model: Trained CNN model
        image: Preprocessed image (1, 28, 28, 1)
        predicted_class: Predicted digit class
        layer_name: Name of convolutional layer to visualize
    
    Returns:
        Base64 encoded heatmap image
    """
    
    try:
        # Get the convolutional layer output and final output
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, predicted_class]
        
        # Get gradients of the loss with respect to conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap = cv2.resize(heatmap, (28, 28))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert original image for overlay
        original_image = image[0, :, :, 0]
        original_image = np.uint8(255 * original_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        
        # Resize for better visibility
        pil_image = pil_image.resize((280, 280), Image.Resampling.NEAREST)
        
        # Convert to base64 for sending to frontend
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    except Exception as e:
        print(f"⚠️  Grad-CAM generation failed: {str(e)}")
        # Return empty image on error
        empty_img = Image.new('RGB', (280, 280), color='black')
        buffered = BytesIO()
        empty_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"