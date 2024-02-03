# Importing neccasary libraries
import cv2
import numpy as np
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import librosa
from pydub import AudioSegment
from pydub.playback import play
import threading
import pyaudio
import wave
from keras.utils import to_categorical
import joblib

# Load saved scaler and encoder from pretrained model file
scaler = joblib.load('fitted_scaler.save') 
encoder = joblib.load('fitted_encoder.save')

# Defining a function for loading the pretrained FacialExpressionModel
def load_facial_expression_model(json_file, weight_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weight_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Defining a function for Feature Extraction
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Create a variable for facial expression model
facial_model = load_facial_expression_model('model_a1.json', 'model_weights1.h5')

# Create a list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create a variable for using haarcascade_frontalface_default.xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up PyAudio for live audio capture
p = pyaudio.PyAudio()

# Global variable to stop both threads when needed
stop_threads = False

# Defining a function to predict emotion from facial model
def predict_facial_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))

        img_pixels = image.img_to_array(face_roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels / 255.0

        predictions = facial_model.predict(img_pixels)
        max_index = np.argmax(predictions)
        emotion = emotion_labels[max_index]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'Facial: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)

# Defining a function to predict emotion from speech model
def predict_speech_emotion(audio_file_path):
    # Create a variable for speech emotion model
    speech_model = load_model('speech_recognition.h5')  

    # Extract MFCC features from the audio file
    mfcc = extract_mfcc(audio_file_path)  

    # Preprocess the MFCC features for the model
    mfcc = np.expand_dims(scaler.transform(np.expand_dims(mfcc, axis=0)), axis=2)

    # Predict emotion using the loaded model
    emotion_probs = speech_model.predict(mfcc)

    # Convert probabilities to one-hot encoded labels
    one_hot_label = to_categorical(np.argmax(emotion_probs, axis=1), num_classes=7)
    emotion_label = encoder.inverse_transform(one_hot_label)[0][0]

    # Print probabilities for each class
    for i, prob in enumerate(emotion_probs[0]):
        print(f"Probability of class {i}: {prob}")

    return emotion_label


# Defining Function to predict emotion from live speech
def predict_live_speech_emotion():
    frames = []
    global stop_threads

    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    print("Recording...")
    while not stop_threads:
        data = stream.read(1024)
        frames.append(data)

    print("Recording done.")
    stream.stop_stream()
    stream.close()

    # Save the recorded audio to a file
    wf = wave.open("live_speech.wav", 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Predict emotion from live speech
    emotion_label = predict_speech_emotion("live_speech.wav")
    print(f'Live Speech: {emotion_label}')
    play(AudioSegment.from_file("live_speech.wav"))  # Play the recorded audio

# Function to process live facial expression and speech
def process_live_emotion():
    global stop_threads
    cap = cv2.VideoCapture(0)

    while not stop_threads:
        ret, frame = cap.read()

        # Predict emotion from facial model
        predict_facial_emotion(frame)

        cv2.imshow('Live Emotion Detection', frame)
        
        # Breaks the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_threads = True
            
    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Run threads for live emotion processing
facial_thread = threading.Thread(target=process_live_emotion)
speech_thread = threading.Thread(target=predict_live_speech_emotion)

facial_thread.start()
speech_thread.start()

facial_thread.join()
speech_thread.join()

# Close PyAudio
p.terminate()