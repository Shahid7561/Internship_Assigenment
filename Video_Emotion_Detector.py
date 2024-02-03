# Importing neccasary libraries
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Defining a function for loading the pretrained FacialExpressionModel
def FacialExpressionModel(json_file,weight_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weight_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Create a variable for model
model = FacialExpressionModel('model_a1.json','model_weights1.h5')

# Create a variable for using haarcascade_frontalface_default.xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a list of emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Using default camera option
cap = cv2.VideoCapture(0)

# Detect faces from video stream
while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))

        # Preprocess the image for the FER model
        img_pixels = image.img_to_array(face_roi)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels / 255.0

        # Predict the emotion
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions)
        emotion = emotion_labels[max_index]

        # Draw the bounding box and emotion text on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('Emotion Detection', frame)

    # Breaks the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()