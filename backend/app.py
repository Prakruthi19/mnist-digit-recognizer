"""
Flask server for MNIST digit recognition
Serves predictions from MLP and CNN models
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import os

# Import our helper functions
from predict import preprocess_image, predict_digit
from gradcam import generate_gradcam

# Load models at startup
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("🚀 Starting MNIST Digit Recognition Server")
print("=" * 60)
# print(f"TensorFlow version: {tf.__version__}")
# print(f"Keras version: {keras.__version__}")
print()

# Load trained models with compile=False to avoid version issues
print("📦 Loading models...")
try:
    # Load without compiling (skips the batch_shape issue)
    mlp_model = keras.models.load_model('../models/saved_models/mlp_model.keras')
    cnn_model = keras.models.load_model('../models/saved_models/cnn_model.keras')
    
    # Recompile models manually
    print("🔧 Recompiling models...")
    mlp_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    cnn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Models loaded and compiled successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print()
    print("💡 Solutions:")
    print("   1. Make sure TensorFlow versions match between training and deployment")
    print("   2. Retrain models with: pip install tensorflow==2.15.0")
    print("   3. Check if model files exist in '../saved_models/'")
    exit(1)

print()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mlp_loaded': mlp_model is not None,
        'cnn_loaded': cnn_model is not None,
        'tensorflow_version': tf.__version__
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Receives image, returns predictions from both models
    """
    try:
        # Get image data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess for MLP (flattened)
        image_mlp = preprocess_image(image, model_type='mlp', debug=False)
        
        # Preprocess for CNN (2D with channel)
        image_cnn = preprocess_image(image, model_type='cnn', debug=False)
        
        # Get predictions from both models
        mlp_prediction = predict_digit(mlp_model, image_mlp, debug=False)
        cnn_prediction = predict_digit(cnn_model, image_cnn, debug=False)
        
        # Generate Grad-CAM heatmap for CNN
        gradcam_image = generate_gradcam(cnn_model, image_cnn, cnn_prediction['digit'])
        
        # Prepare response
        response = {
            'mlp': mlp_prediction,
            'cnn': cnn_prediction,
            'gradcam': gradcam_image  # Base64 encoded heatmap
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_models():
    """
    Compare MLP vs CNN performance
    Returns detailed comparison statistics
    """
    try:
        data = request.get_json()
        image_data = data['image']
        
        # Decode image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get predictions
        image_mlp = preprocess_image(image, model_type='mlp', debug=False)
        image_cnn = preprocess_image(image, model_type='cnn', debug=False)
        
        mlp_result = predict_digit(mlp_model, image_mlp, debug=False)
        cnn_result = predict_digit(cnn_model, image_cnn, debug=False)
        
        # Calculate comparison metrics
        comparison = {
            'agreement': mlp_result['digit'] == cnn_result['digit'],
            'mlp_confidence': mlp_result['confidence'],
            'cnn_confidence': cnn_result['confidence'],
            'confidence_difference': abs(cnn_result['confidence'] - mlp_result['confidence']),
            'winner': 'CNN' if cnn_result['confidence'] > mlp_result['confidence'] else 'MLP'
        }
        
        return jsonify({
            'mlp': mlp_result,
            'cnn': cnn_result,
            'comparison': comparison
        })
    
    except Exception as e:
        print(f"❌ Error during comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================
# START SERVER
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("✅ Server is ready!")
    print("🌐 Open http://localhost:5000 in your browser")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)