import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
from custom_losses import OrthogonalRegularizer, FocalLoss, CombinedLoss  # Assuming custom_losses.py has the losses

app = Flask(__name__)

# Load the model with custom loss functions
model = tf.keras.models.load_model(
    'model/model_87.h5',
    custom_objects={
        'combined_loss_2': CombinedLoss,
        'OrthogonalRegularizer': OrthogonalRegularizer,
        'FocalLoss': FocalLoss
    }
)

# Normalize image function (adjust this based on your model's input preprocessing)
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to the input shape the model expects
    image = np.array(image)  # Convert to numpy array
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API route for OCR text prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read and preprocess the image
        image = Image.open(BytesIO(file.read()))
        image = preprocess_image(image)
        
        # Get model predictions
        predictions = model.predict(image)
        
        # Post-process predictions (adjust this based on your output format)
        predicted_class = np.argmax(predictions, axis=-1)
        recognized_text = str(predicted_class[0])  # Modify if your model predicts more complex text
        
        return jsonify({'predicted_text': recognized_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
