import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(_name_)
CORS(app, origins="*")

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_PATH'] = os.path.join('src', 'cnn', 'model_Fashion.h5')

model = load_model(app.config['files'])

# Diccionario de nombres de clases
class_names = {
    0: 'T-shirt/camiseta',
    1: 'Trouser/Pantalon',
    2: 'Pullover/Buso',
    3: 'Dress/Vestido',
    4: 'Coat/Abrigo',
    5: 'Sandal/Sanalia',
    6: 'Shirt/Camisa',
    7: 'Sneaker/Zapato',
    8: 'Bag/Bolso',
    9: 'Ankle boot/Botín'
}

@app.route("/")
def show_homepage():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({"message": "error", "error": "No image part"}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({"message": "error", "error": "No selected image"}), 400
    
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32')
        img /= 255

        prediction = model.predict(img)
        predicted_class = int(prediction.argmax())
        class_name = class_names.get(predicted_class, 'Clase Desconocida')

        return jsonify({"message": "ok", "prediction": class_name})

    except Exception as e:
        return jsonify({"message": "error", "error": str(e)}), 500
    
    finally:
        os.remove(filepath)  

if _name_ == '_main_':
    app.run()
