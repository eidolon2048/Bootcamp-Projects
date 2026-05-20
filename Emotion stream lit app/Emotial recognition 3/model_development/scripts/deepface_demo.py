import cv2
import matplotlib.pyplot as plt
import pandas as pd
from deepface import DeepFace


frameWidth = 1280
frameHeight = 720
cap = cv2.VideoCapture(1) # or set 0
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
cap.set(cv2.CAP_PROP_FPS, 60)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while cap.isOpened():
    success, img = cap.read()

    if success:

        ret,frame = cap.read()   #retiving

        result = DeepFace.analyze(frame, actions = ["emotion"], enforce_detection=False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw rectangle across the face
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(img,(x, y), (x+w, y+h), (0, 255, 0), 2)

        # Add lable of emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        #use putText() method for inserting text on video
        cv2.putText(frame, result[0]["dominant_emotion"],
                    (10, 70), font, 1.7,
                    (100, 0, 200), 2,
                    cv2.LINE_8);
        
        cv2.imshow("Dominant Emotion", frame)
        k = cv2.waitKey(100) & 0xFF 
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()