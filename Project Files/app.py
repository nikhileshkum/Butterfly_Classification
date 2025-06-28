# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import pickle
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/butterfly_classification_model.h5'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
CLASS_NAMES_PATH = 'models/class_names.pkl'
IMG_SIZE = 128

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and label data once at startup
model = load_model(MODEL_PATH)
with open(LABEL_ENCODER_PATH, 'rb') as f:
    label_encoder = pickle.load(f)
with open(CLASS_NAMES_PATH, 'rb') as f:
    class_names = pickle.load(f)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/input')
def input_page():
    """Image upload page"""
    return render_template('input.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    """Handle image upload and classification"""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        processed_img = preprocess_image(filepath)
        predictions = model.predict(processed_img)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[0][predicted_idx])

        # Get top 5 predictions
        top5_indices = np.argsort(predictions[0])[::-1][:5]
        top5_classes = [class_names[i] for i in top5_indices]
        top5_confidences = [float(predictions[0][i]) for i in top5_indices]
        top5_predictions = list(zip(top5_classes, top5_confidences))

        return render_template('output.html',
                               filename=filename,
                               prediction=predicted_class,
                               confidence=confidence,
                               top_predictions=top5_predictions)

    return redirect(request.url)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)