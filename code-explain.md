## <p align="center"> Here's an explanation of the code:</p>

```md
import cv2
from gtts import gTTS
import os
import numpy as np
from tensorflow.keras.models import load_model
```

<p align="justify">
  These lines import the necessary libraries and modules for the code, including OpenCV (cv2), gTTS (Google Text-to-Speech) for generating speech output, os for system operations, numpy for array operations, and load_model from Keras for loading the pre-trained facial expression detection model.
</p>

## <p align="center"></p>


```md
cascade_path = 'haarcascade_frontalface_alt.xml'
model_path = 'facial_expression_model.h5'
```

<p align="justify">
  These lines define the paths to the Haar cascade XML file for face detection (cascade_path) and the pre-trained facial expression detection model (model_path). Make sure these files exist in the same directory as the Python script.
</p>

## <p align="center"></p>


```md
face_cascade = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)
```

<p align="justify">
  These lines load the Haar cascade XML file using CascadeClassifier from OpenCV and load the pre-trained facial expression detection model using load_model from Keras.
</p>

## <p align="center"></p>


```md
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Neutral']
```

<p align="justify">
  This line defines a list of emotion labels. These labels correspond to the emotions predicted by the model and will be used to display the detected emotion on the frame.
</p>

## <p align="center"></p>


```md
cam = cv2.VideoCapture(0)
```

<p align="justify">
  This line initializes the video capture object to capture frames from the default camera ('0').
</p>

## <p align="center"></p>


```md
while True:
    ret, frame = cam.read()
    if not ret:
        break
```

<p align="justify">
  These lines read frames from the video capture object (cam.read()) and store the current frame in the frame variable. The loop continues as long as frames are being successfully read from the camera (ret indicates the success of the read operation). If no frames are read, the loop breaks.
</p>

## <p align="center"></p>


```md
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

<p align="justify">
  This line converts the current frame (frame) from BGR color format to grayscale using cv2.cvtColor().
</p>

## <p align="center"></p>


```md
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

<p align="justify">
  This line detects faces in the grayscale frame (gray) using the Haar cascade classifier (face_cascade). The detectMultiScale() function returns a list of rectangles representing the bounding boxes of the detected faces.
</p>

## <p align="center"></p>


```md
for (x, y, w, h) in faces:
    roi = gray[y:y + h, x:x + w]
    roi = cv2.resize(roi, (48, 48))
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)
    preds = model.predict(roi)[0]
    emotion_label = emotion_labels[np.argmax(preds)]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    language = "en"
    output = gTTS(text=emotion_label, lang=language, slow=False)
    output.save("output.mp3")
    os.system("mpg123 output.mp3")

```

<p align="justify">
  These lines iterate over the detected faces and perform the following operations:
  - Extract the face region of interest (ROI) from the grayscale frame.
  - Resize the ROI to match the input size expected by the facial expression detection model (48x48 pixels).
  - Expand the dimensions of the ROI array to match the input shape expected by the model.
  - Perform emotion prediction on the ROI using the pre-trained model.
  - Get the predicted emotion label corresponding to the highest predicted emotion probability.
  - Draw a rectangle around the face and display the emotion label on the frame.
  - Generate speech output for the emotion label using gTTS, save it as an MP3 file (output.mp3), and play it using mpg123 command-line tool.
</p>

## <p align="center"></p>


```md
cv2.imshow('Facial Expression Detection', frame)
if cv2.waitKey(1) == 27:
    break
```

<p align="justify">
  These lines display the resulting frame with bounding boxes and emotion labels using cv2.imshow(). The imshow() function takes the frame and a window title as input. The loop continues until the ESC key (ASCII code 27) is pressed, at which point the loop breaks.
</p>

## <p align="center"></p>


```md
cam.release()
cv2.destroyAllWindows()
```

<p align="justify">
  These lines release the video capture object (cam.release()) and close all OpenCV windows (cv2.destroyAllWindows()).
</p>

## <p align="center"></p>

<br>

#### <p align="center"> Copyright © Porinita Hoque (ID : 1711204042) </p>
