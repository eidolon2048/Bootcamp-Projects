# Emotion Recognition Streamlit App

Streamlit application for facial emotion recognition from uploaded images, uploaded videos, and live webcam input.

The app detects faces with OpenCV, preprocesses each detected face to `48x48` grayscale, and predicts one of five emotion classes using a trained Keras/TensorFlow CNN model.

## For Local Run
```
conda activate streamlit_environment
pip install -r requirements.txt
streamlit run app.py
```
## Or use ready link if doesn't work
Live app: [https://emotial-recognition.streamlit.app](https://emotial-recognition.streamlit.app)

## Features

- Image upload emotion detection
- Video upload emotion detection with frame sampling
- Live webcam emotion detection
- Face bounding boxes with predicted emotion labels
- Confidence scores and class probability table

## Emotion Classes

The model predicts:

- angry
- happy
- neutral
- sad
- surprise

## Project Files

```text
app.py
  Main Streamlit application.

requirements.txt
  Python dependencies required by the app.

models/
  Trained Keras model files used for prediction:
  - emotion_model4.json
  - emotion_model4.h5
