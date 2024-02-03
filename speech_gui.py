# Importing neccasary libraries
import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from keras.models import load_model
import joblib
from keras.utils import to_categorical
from pydub import AudioSegment
from pydub.playback import play

# Load the pre-trained model
model = load_model('speech_recognition.h5')

scaler = joblib.load('fitted_scaler.save') 
encoder = joblib.load('fitted_encoder.save')

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

# Defining the function to browse file
def browse_file():
    global file_path  # Declare file_path as global
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
    if file_path:
        emotion_label = predict_emotion(file_path)
        result_label.configure(text=f"Predicted Emotion: {emotion_label}")

# Create the main GUI window
root = tk.Tk()
root.title("Speech Emotion Recognition")

root.geometry("400x200")

# Create GUI components
file_button = tk.Button(root, text="Select Audio File", command=browse_file)
play_button = tk.Button(root, text="Play Audio File", command=lambda: play_audio(file_path))
result_label = tk.Label(root, text="Predicted Emotion: ")

# Pack the components into the window
file_button.pack(pady=10)
play_button.pack(pady=10)
result_label.pack(pady=10)

# Run the GUI application
root.mainloop()