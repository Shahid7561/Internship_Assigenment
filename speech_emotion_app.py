from flask import Flask, render_template, request
import librosa
import numpy as np
from keras.models import load_model
import joblib
from keras.utils import to_categorical
from pydub import AudioSegment
from pydub.playback import play
import os

app = Flask(__name__)

# Load the pre-trained model, scaler, and encoder
model = load_model('speech_recognition.h5')
scaler = joblib.load('fitted_scaler.save')
encoder = joblib.load('fitted_encoder.save')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Defining function to predict_emotion
def predict_emotion(file_path):
    mfcc = extract_mfcc(file_path)
    mfcc = np.expand_dims(scaler.transform(np.expand_dims(mfcc, axis=0)), axis=2)
    emotion_probs = model.predict(mfcc)

    # Convert probabilities to one-hot encoded labels
    one_hot_label = to_categorical(np.argmax(emotion_probs, axis=1), num_classes=7)
    emotion_label = encoder.inverse_transform(one_hot_label)[0][0]
    for i, prob in enumerate(emotion_probs):
        print(f"Probability of class {i}: {prob}")
    return emotion_label

# Defining the function to play selected audio
def play_audio(file_path):
    sound = AudioSegment.from_file(file_path)
    play(sound)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('speech_index.html', result="No file uploaded")
        file = request.files['file']
        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('speech_index.html', result="No file selected")
        if file and allowed_file(file.filename):
            # Create the 'uploads' directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            emotion_label = predict_emotion(file_path)
            return render_template('speech_index.html', result=f"Predicted Emotion: {emotion_label}", audio_path=file_path)
    return render_template('speech_index.html', result=None)
