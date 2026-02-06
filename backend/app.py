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

# Load trained models
print("📦 Loading models...")
mlp_model = keras.models.load_model('../saved_models/mlp_model_fixed.keras')
cnn_model = keras.models.load_model('../saved_models/cnn_model_fixed.keras')

print("✅ Models loaded successfully!")
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
        'cnn_loaded': cnn_model is not None
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
        image_mlp = preprocess_image(image, model_type='mlp')
        
        # Preprocess for CNN (2D with channel)
        image_cnn = preprocess_image(image, model_type='cnn')
        
        # Get predictions from both models
        mlp_prediction = predict_digit(mlp_model, image_mlp)
        cnn_prediction = predict_digit(cnn_model, image_cnn)
        
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
        image_mlp = preprocess_image(image, model_type='mlp')
        image_cnn = preprocess_image(image, model_type='cnn')
        
        mlp_result = predict_digit(mlp_model, image_mlp)
        cnn_result = predict_digit(cnn_model, image_cnn)
        
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