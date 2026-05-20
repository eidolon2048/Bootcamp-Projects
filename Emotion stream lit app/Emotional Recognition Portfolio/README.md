# Emotional Recognition

Portfolio project for facial emotion recognition using a custom CNN model and a Streamlit application.

Live app:

```text
https://emotial-recognition.streamlit.app
```

## Project Structure

```text
1. Data/
  Dataset notes and expected FER 2013 folder layout.

2. Model 1 - DeepFace Baseline/
  Baseline emotion-recognition script using the DeepFace library.

3. Model 2 - Custom CNN/
  Training notebook, OpenCV/Keras video script, and trained model artifacts.

4. Streamlit App/
  Runnable Streamlit app for image upload, video upload, and live camera emotion recognition.

5. Presentation/
  Project presentation slides.
```

## Goal

The goal of this project is to detect faces and classify facial emotions from images and video. The final application predicts five emotions:

- angry
- happy
- neutral
- sad
- surprise

## Approach

1. Use FER 2013-style facial expression data for model development.
2. Compare a DeepFace baseline with a custom CNN model.
3. Use OpenCV Haar cascade face detection to locate faces.
4. Preprocess detected faces to grayscale `48x48` inputs.
5. Predict emotion with a trained Keras/TensorFlow model.
6. Deploy the workflow as a Streamlit app.

## Run The Streamlit App Locally

```bash
cd "4. Streamlit App"
conda activate streamlit_environment
streamlit run app.py
```

## Deployment

For Streamlit Community Cloud, deploy only the clean app files:

```text
app.py
requirements.txt
models/
```

The full portfolio folder includes notebooks, scripts, model development files, and presentation materials.
