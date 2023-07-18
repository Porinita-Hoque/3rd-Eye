import cv2
from gtts import gTTS
import os
import numpy as np
from tensorflow.keras.models import load_model

# Define the paths to the Haar cascade XML file and the pre-trained facial expression detection model
cascade_path = 'haarcascade_frontalface_alt.xml'
model_path = 'facial_expression_model.h5'

# Load the Haar cascade XML file and the pre-trained facial expression detection model
face_cascade = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)

# Define the emotions labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the video capture
cam = cv2.VideoCapture(0)

while True:
    # Read the current frame from the video capture
    ret, frame = cam.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        
        # Perform emotion prediction on the face ROI
        preds = model.predict(roi)[0]
        emotion_label = emotion_labels[np.argmax(preds)]
        
        # Draw a rectangle around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Generate speech output for the emotion label
        language = "en"
        output = gTTS(text=emotion_label, lang=language, slow=False)
        output.save("output.mp3")
        os.system("mpg123 output.mp3")
    
    # Display the resulting frame
    cv2.imshow('Facial Expression Detection', frame)
    
    # Check for key press (ESC to exit)
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close all windows
cam.release()
cv2.destroyAllWindows()
