# Importing neccasary libraries
import cv2
import dlib
from scipy.spatial import distance

# Calculate distance of eye
def calculate_EYE(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio

# Using default camera option
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()

# Create a variable for using shape_predictor_68_face_landmarks.dat file
dlib_facelandmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Detect eyes from video stream
while True:
    _,frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmarks(gray,face)
        leftEye = []
        rightEYE = []

        # detect leftEye from face
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n==41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x 
            y2 = face_landmarks.part(next_point).y 
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        # detect rightEYE from face
        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEYE.append((x,y))
            next_point = n+1
            if n==47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x 
            y2 = face_landmarks.part(next_point).y 
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        left_eye = calculate_EYE(leftEye)
        right_eye = calculate_EYE(rightEYE)

        EYE = (left_eye+right_eye)/2
        EYE = round(EYE,2)
        
        # Pick random value of EYE based on trail and error,based on that value display text
        if EYE<0.26:
            cv2.putText(frame,"DROWSY",(20,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            cv2.putText(frame,"Are You Sleepy?",(50,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)

    cv2.imshow("Are You Sleepy?",frame)

    # Breaks the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release the webcam and close the OpenCV window    
cap.release()
cv2.destroyAllWindows()