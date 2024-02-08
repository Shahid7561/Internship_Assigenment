from flask import Flask, render_template, Response
import cv2
import dlib
from scipy.spatial import distance

app = Flask(__name__)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_EYE(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_aspect_ratio = (A + B) / (2.0 * C)
    return eye_aspect_ratio

def detect_drowsiness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmarks(gray, face)
        leftEye = []
        rightEYE = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEYE.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_eye = calculate_EYE(leftEye)
        right_eye = calculate_EYE(rightEYE)

        EYE = (left_eye + right_eye) / 2
        EYE = round(EYE, 2)

        if EYE < 0.26:
            cv2.putText(frame, "DROWSY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
            cv2.putText(frame, "Are You Sleepy?", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detect_drowsiness(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('drowsiness_index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
