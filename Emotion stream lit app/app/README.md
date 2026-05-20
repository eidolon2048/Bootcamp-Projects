# Emotion Recognition Streamlit App

Streamlit application for facial emotion recognition in live webcam video, uploaded images, and uploaded videos.

## Run From Finder

Double-click:

```text
run_app.command
```

macOS may ask for permission the first time. Keep the terminal window open while the app is running.

## Run From VS Code

Use Command Palette:

```text
Tasks: Run Task
```

Then choose:

```text
Run Emotion Streamlit App
```

To verify the environment, choose:

```text
Check Emotion App Environment
```

## Run From Terminal

```bash
conda activate streamlit_environment
pip install -r requirements.txt
streamlit run app.py
```

Do not run this app with `python app.py`; Streamlit apps must be started with `streamlit run app.py`.

## Model

The app uses the trained Keras model copied into `models/`:

- `models/emotion_model4.json`
- `models/emotion_model4.h5`

The model predicts five emotion classes: `angry`, `happy`, `neutral`, `sad`, and `surprise`.
