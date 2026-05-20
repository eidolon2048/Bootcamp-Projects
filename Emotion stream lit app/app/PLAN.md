# Emotion Recognition Streamlit App Plan

## Reference Rules

- Do not change `Emotional recognition 1`.
- Do not change `Emotional_recognition`.
- Use `Emotional recognition 1` as the main reference for model code and project information.
- Use `Emotional_recognition/streamlit_app` as the Streamlit app reference.
- Build the new application only inside `app`.

## Current Findings

- The target `app` folder is empty.
- The trained model is stored as JSON architecture plus H5 weights.
- The model expects grayscale face crops shaped as `48x48x1`.
- The model output has five classes:
  - `angry`
  - `happy`
  - `neutral`
  - `sad`
  - `surprise`
- The reference video code uses OpenCV Haar cascade face detection.
- The reference Streamlit app already starts a `streamlit_webrtc` webcam stream, but its emotion prediction code is incomplete/commented out.
- Existing model paths are hard-coded absolute paths and must be replaced with app-relative paths.

## Implementation Plan

1. Create a clean Streamlit project inside `app`.
   - `app.py`
   - `requirements.txt`
   - `models/emotion_model4.json`
   - `models/emotion_model4.h5`
   - `README.md`

2. Copy model assets into `app/models`.
   - Copy from the reference folders.
   - Do not edit the reference folders.
   - Use app-relative paths in the application.

3. Build reusable emotion recognition logic.
   - Load the Keras model once with `st.cache_resource`.
   - Load Haar cascade from `cv2.data.haarcascades`.
   - Detect faces in grayscale frames.
   - Resize each face region to `48x48`.
   - Normalize pixel values to `0-1`.
   - Reshape to `(1, 48, 48, 1)`.
   - Return emotion label, confidence score, and class probabilities.

4. Build the Streamlit interface.
   - Provide a compact app title and controls.
   - Include modes for live webcam, uploaded image, and uploaded video.
   - Show annotated frames/images with face boxes, emotion labels, and confidence.
   - Show prediction details when useful.

5. Implement live webcam detection.
   - Use `streamlit_webrtc`.
   - Disable audio.
   - Process each video frame.
   - Draw face boxes and emotion labels directly on the stream.

6. Implement uploaded image detection.
   - Support common image formats.
   - Detect all visible faces.
   - Display the annotated image.
   - Show per-face prediction results.

7. Implement uploaded video detection.
   - Support common video formats where OpenCV can decode them.
   - Process sampled frames for performance.
   - Display an annotated preview.
   - Avoid expensive full-video export unless approved later.

8. Add validation and error handling.
   - Handle missing model files.
   - Handle missing or unavailable webcam.
   - Show a clear message when no face is detected.
   - Avoid crashing on unsupported media files.

9. Verify the app.
   - Run `streamlit run app.py` from inside `app`.
   - Confirm the model loads.
   - Confirm uploaded image prediction works.
   - Confirm the webcam component starts.
   - Confirm imports and syntax are valid.
