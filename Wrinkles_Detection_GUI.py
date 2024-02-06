# Importing neccasary libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from keras.models import load_model
from PIL import Image, ImageTk
import numpy as np
import cv2

# Defining a function for loading the pretrained WrinklesDetectionModel
def WrinklesDetectionModel(model_path):
    model = load_model(model_path)
    return model

# Create the main GUI window
top = tk.Tk()
top.geometry('800x600')
top.title('Wrinkles Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

# Create a variable for using haarcascade_frontalface_default.xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create a variable for model
model = WrinklesDetectionModel('Wrinkles_Detection.h5')  


# Defining function to detect wrinkles on face
def Detect(file_path):
    global label1

    image = cv2.imread(file_path)
    
    # Convert the frame to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces from selected image
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

    try:
        if len(faces) == 0:
            raise ValueError("No faces detected in the image.")

        for (x, y, w, h) in faces:
            face_crop = gray_image[y:y+h, x:x+w]
            roi = cv2.resize(face_crop, (48, 48))
            roi = np.expand_dims(roi, axis=-1)  # Channel dimension
            roi = np.expand_dims(roi, axis=0)   # Batch dimension
            roi = np.repeat(roi, 3, axis=-1)    # Repeat grayscale channel to create RGB image
            roi = roi / 255.0  # Normalize pixel values to the range [0, 1]

            # Make predictions
            prediction = model.predict(roi)

            # Interpret the result
            result_text = "Wrinkles Detected!" if prediction > 0.5 else "No Wrinkles Detected!"
            label1.configure(foreground='#011638', text=result_text)

    except Exception as e:
        print(f"Error during detection: {str(e)}")
        label1.configure(foreground='#011638', text="Unable to detect")


# Defining function for show detect button 
def show_detect_button(file_path):
    detect_button = Button(top, text='Detect Wrinkles', command=lambda: Detect(file_path), padx=10, pady=5)
    detect_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)


# Defining function to upload image in gui
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.3), (top.winfo_height()/2.3)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(e)

upload = Button(top,text = "Upload Image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156',foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand='True')
label1.pack(side='bottom',expand='True')
heading = Label(top,text='Wrinkles Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()

# Run the GUI application
top.mainloop()