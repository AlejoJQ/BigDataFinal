import flask
from flask import request, render_template
import numpy as np
from keras.models import load_model
import cv2  
from flask import redirect
import json
import os
from werkzeug.utils import secure_filename

app = flask.Flask(__name__, template_folder='.')
model = load_model("model_Fashion.h5")

with open('clases.json', 'r') as json_file:
    classes_data = json.load(json_file)

classes = classes_data.get('clases', [])
model.classes_ = classes

initial_image_path = ""

def predict_animal(filepath):
    
    img = cv2.imread(filepath)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (28, 28))

    gray = gray.astype('float32') / 255

    gray = gray.reshape(1, 28, 28, 1)

    prediction = model.predict(gray)

    class_id = np.argmax(prediction)

    class_name = model.classes_[class_id]

    return class_name

@app.route("/", methods=["GET", "POST"])
def index():
    class_name = None
    selected_image_path = initial_image_path

    if request.method == "POST":
        image = request.files["image"]
        if image.filename != '':
            filename = secure_filename(image.filename)  
            selected_image_path = os.path.join("static", filename)
            image.save(selected_image_path)
            class_name = predict_animal(selected_image_path)

    return render_template("index.html", class_name=class_name, selected_image_path=selected_image_path)


if __name__ == "__main__":
    app.run(debug=True)
