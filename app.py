from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
app = Flask(__name__)
model_path = 'vgg16_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print(" Model file not found.")
    model = None
labels = {0: 'biodegradable', 1: 'recyclable', 2: 'trash'}
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please check 'vgg16_model.h5'."
    if 'file' not in request.files:
        return "No file uploaded."
    file = request.files['file']
    file_path = os.path.join('static', file.filename)
    file.save(file_path)
    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        return f"Error processing image: {e}"
    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        result = labels[predicted_class]
    except Exception as e:
        return f"Prediction error: {e}"
    return render_template('result.html', prediction=result, image_path=file_path)
if __name__ == '__main__':
    app.run(debug=True)
