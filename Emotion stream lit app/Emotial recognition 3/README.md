# Emotion Recognition 3

This project keeps model development and the runnable Streamlit application in separate folders.

## Structure

```text
model_development/
  notebooks/       Training and experiment notebooks.
  scripts/         Reference scripts for DeepFace and OpenCV/Keras video detection.
  trained_model/   Trained Keras model architecture and weights.
  data/            Dataset notes. Raw dataset files are not committed.

streamlit_app/
  app.py                 Streamlit emotion recognition app.
  models/                Model files used by the app.
  run_app.command        macOS double-click launcher.
  check_environment.py   Runtime dependency and model check.
  requirements.txt       Python package requirements.
```

## Run The App

From Finder, double-click:

```text
streamlit_app/run_app.command
```

From VS Code, use:

```text
Tasks: Run Task
```

Then choose:

```text
Run Emotion Streamlit App
```

From terminal:

```bash
cd streamlit_app
conda activate streamlit_environment
streamlit run app.py
```

## Model

The app uses the five-class Keras model:

- `angry`
- `happy`
- `neutral`
- `sad`
- `surprise`

The model expects grayscale face crops shaped as `48x48x1`.
