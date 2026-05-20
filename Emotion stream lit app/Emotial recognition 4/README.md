# Emotion Recognition Streamlit App

Deployment-ready Streamlit app for facial emotion recognition from webcam video, uploaded images, and uploaded video previews.

## Files

```text
app.py
requirements.txt
models/
  emotion_model4.json
  emotion_model4.h5
```

## Deploy On Streamlit Community Cloud

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy.

## Notes

- The uploaded image and uploaded video modes are the most reliable demo paths.
- The live camera mode uses `streamlit-webrtc`; it requires browser camera permission and HTTPS. Streamlit Cloud provides HTTPS.
- The model predicts five classes: `angry`, `happy`, `neutral`, `sad`, and `surprise`.
