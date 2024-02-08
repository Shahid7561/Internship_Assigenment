from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('video_index.html') 

def generate_frames():
    cap = cv2.VideoCapture(0)
    model = FacialExpressionModel('model_a1.json', 'model_weights1.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (48, 48))
            img_pixels = image.img_to_array(face_roi)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels / 255.0

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions)
            emotion = emotion_labels[max_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2,
                        cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def FacialExpressionModel(json_file, weight_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weight_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    app.run(debug=True)
