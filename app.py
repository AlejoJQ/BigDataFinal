import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
from flask_cors import CORS

app = Flask(_name_)
cors = CORS(app,origins="*")
app.config['files'] = os.path.join(os.path.dirname(_file_),'src','files')

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

def save_image(image) -> str:
  filename = secure_filename(image.filename)
  filepath = os.path.join(app.config['files'], filename)
  image.save(filepath)

  return filepath

def predict_image(image_file):
  try:
    filepath = save_image(image_file)
    model_path = os.path.join(os.path.dirname(_file_), 'src','cnn','model_Fashion.h5')
    model = load_model(model_path)

    img = cv2.imread(filepath)
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])

    gray = gray.reshape(1, 28, 28, 1)
    gray /= 255

    prediction = model.predict(gray.reshape(1, 28, 28, 1))

    predicted_class = int(prediction.argmax())

    class_name = class_names.get(predicted_class, 'Clase Desconocida')

    return jsonify({
      "message": "ok",
      "prediction": class_name
    })

  except Exception as e:
    return jsonify({
      "message": "error",
      "error": str(e)
    }), 500


@app.get("/")
def show_homepage():
  return render_template('index.html')


@app.post("/predict")
def predict_image():
  image_file = request.files['image']
  return predict_image(image_file)


if _name_ == '_main_':
  app.run(debug=true)
