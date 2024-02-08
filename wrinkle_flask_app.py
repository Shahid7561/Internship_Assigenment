from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

app = Flask(__name__)

# Load the Wrinkles Detection model
model = load_model('Wrinkles_Detection.h5')

# Create a variable for using haarcascade_frontalface_default.xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_wrinkles(image_path):
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        if len(faces) == 0:
            raise ValueError("No faces detected in the image.")

        for (x, y, w, h) in faces:
            face_crop = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(face_crop, (48, 48))
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            roi = np.repeat(roi, 3, axis=-1)
            roi = roi / 255.0

            prediction = model.predict(roi)

            result_text = "Wrinkles Detected!" if prediction > 0.5 else "No Wrinkles Detected!"
            return result_text

    except Exception as e:
        print(f"Error during detection: {str(e)}")
        return "Unable to detect"

@app.route('/')
def index():
    return render_template('wrinkles_index.html')

@app.route('/detect_wrinkles', methods=['POST'])
def detect_wrinkles_route():
    try:
        if 'file' not in request.files:
            return render_template('wrinkles_result.html', result="No file uploaded")

        file = request.files['file']

        if file.filename == '':
            return render_template('wrinkles_result.html', result="No file selected")

        image_path = 'temp_image.jpg'
        file.save(image_path)

        result_text = detect_wrinkles(image_path)
        return render_template('wrinkles_result.html', result=result_text)

    except Exception as e:
        print(e)
        return render_template('wrinkles_result.html', result="Error during detection")

if __name__ == "__main__":
    app.run(debug=True)
