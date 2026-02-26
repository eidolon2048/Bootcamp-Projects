import numpy as np
import cv2
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array

emotion_dict = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}

# --- Load model ---
json_path = "/Users/roma/Desktop/Emotional_recognition/streamlit_app/emotion_model4.json"
weights_path = "/Users/roma/Desktop/Emotional_recognition/streamlit_app/emotion_model4.h5"

with open(json_path, "r") as f:
    loaded_model_json = f.read()

classifier = model_from_json(loaded_model_json)
classifier.load_weights(weights_path)

# --- Camera setup ---
frameWidth, frameHeight = 1280, 720
# frameWidth, frameHeight = 1920, 1080
# frameWidth, frameHeight = 2048, 1080

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_FPS, 60)

# --- Face detector ---
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ROI
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Normalize + shape: (1, 48, 48, 1)
        roi = roi_gray.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=-1)      # add channel
        roi = np.expand_dims(roi, axis=0)       # add batch

        pred = classifier.predict(roi, verbose=0)[0]
        maxindex = int(np.argmax(pred))
        label = emotion_dict.get(maxindex, "unknown")
        conf = float(np.max(pred))

        # Put label above rectangle
        label_text = f"{label} ({conf:.2f})"
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 0, 200), 2, cv2.LINE_AA)

    cv2.imshow("Your Emotion", frame)

    # ESC to quit
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()