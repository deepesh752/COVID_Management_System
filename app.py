import numpy
from werkzeug.utils import secure_filename
from PIL import Image as im
import os
import cv2
from flask import Flask, render_template, request, jsonify
import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
srcnn_model = keras.models.load_model('models/SRCNN_model.h5')
srcnn_model.load_weights("weights/3051crop_weight_200.h5")
srcnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

model = keras.models.load_model('models/detection_model.h5')


# define necessary image processing functions
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


# super resolution function
def resolve(image_path):
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)

    # preprocess the image with modcrop
    degraded = modcrop(degraded, 3)

    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)

    # perform super-resolution with srcnn
    pre = srcnn_model.predict(temp, batch_size=1)
    resolve_img = cv2.cvtColor(pre, cv2.COLOR_YCrCb2BGR)

    # return images and scores
    return resolve_img


def model_predict(img_path, model):
    print("Model Predict")
    # Super Resolve Image
    sr_img = resolve(img_path)
    med_img = im.fromarray(sr_img, 'RGB')

    basepath = os.path.dirname(__file__)
    med_path = os.path.join(
        basepath, 'uploads', 'mediator.png')
    med_img.save(med_path)

    img = image.load_img(med_path, target_size=(224, 224))
    os.remove(med_path)
    # Preprocessing the image
    x = image.img_to_array(img, dtype='double')

    x = x / 255
    x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)

    preds = model.predict_classes(x)
    print("preds")
    if preds == 0:
        return "Negative"
    elif preds == 1:
        return "Positive"


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        print("Make Predictions")
        pred = model_predict(file_path, model)
        os.remove(file_path)
        print("PREDS", pred)
        return pred
    return None


@app.route('/cases', methods=['GET', 'POST'])
def cases():
    return render_template('cases.html')


@app.route('/info', methods=['GET', 'POST'])
def info():
    return render_template('info.html')


@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)
