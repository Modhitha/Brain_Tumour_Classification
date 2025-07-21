from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2

app = Flask(__name__)

# Load both models
vanilla_model = load_model("/Users/koti/Desktop/Brain_Tumor_Classification_AI/models/vanilla_cnn.keras")
vgg_model = load_model("/Users/koti/Desktop/Brain_Tumor_Classification_AI/models/vgg16_cnn.keras")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
img_size = 150

def preprocess_image(filepath, model_type="vanilla"):
    if model_type == "vanilla":
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape(1, img_size, img_size, 1) / 255.0
    else:  # For VGG16 (RGB)
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_size, img_size))
        img = img.reshape(1, img_size, img_size, 3) / 255.0
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file uploaded")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Choose model (default: vanilla)
    model_choice = request.form.get("model", "vanilla")  # add a dropdown in HTML for this
    model = vanilla_model if model_choice == "vanilla" else vgg_model

    # Preprocess and predict
    img = preprocess_image(filepath, model_type=model_choice)
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return render_template('index.html',
                           prediction=predicted_class,
                           confidence=confidence,
                           img_path=filepath,
                           model_used=model_choice)

if __name__ == '__main__':
    app.run(debug=True)
