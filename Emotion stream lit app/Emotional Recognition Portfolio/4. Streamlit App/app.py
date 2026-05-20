import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from tensorflow.keras.models import model_from_json


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_JSON_PATH = MODEL_DIR / "emotion_model4.json"
MODEL_WEIGHTS_PATH = MODEL_DIR / "emotion_model4.h5"

EMOTIONS = {
    0: "angry",
    1: "happy",
    2: "neutral",
    3: "sad",
    4: "surprise",
}

EMOTION_COLORS = {
    "angry": (68, 68, 220),
    "happy": (50, 180, 70),
    "neutral": (200, 160, 40),
    "sad": (180, 90, 40),
    "surprise": (170, 70, 180),
}

CACHE_RESOURCE = getattr(st, "cache_resource", st.cache)


@CACHE_RESOURCE
def load_emotion_model():
    if not MODEL_JSON_PATH.exists() or not MODEL_WEIGHTS_PATH.exists():
        missing = [
            str(path.relative_to(APP_DIR))
            for path in (MODEL_JSON_PATH, MODEL_WEIGHTS_PATH)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing model file(s): {', '.join(missing)}")

    model_json = MODEL_JSON_PATH.read_text()
    model = model_from_json(model_json)
    model.load_weights(str(MODEL_WEIGHTS_PATH))
    return model


@CACHE_RESOURCE
def load_face_detector():
    cascade_filename = "haarcascade_frontalface_default.xml"
    cascade_candidates = []

    if hasattr(cv2, "data") and getattr(cv2.data, "haarcascades", None):
        cascade_candidates.append(Path(cv2.data.haarcascades) / cascade_filename)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cascade_candidates.append(
            Path(conda_prefix) / "share" / "opencv4" / "haarcascades" / cascade_filename
        )

    cascade_candidates.append(MODEL_DIR / cascade_filename)

    for cascade_path in cascade_candidates:
        if cascade_path.exists():
            detector = cv2.CascadeClassifier(str(cascade_path))
            if not detector.empty():
                return detector

    checked_paths = ", ".join(str(path) for path in cascade_candidates)
    raise FileNotFoundError(f"OpenCV Haar cascade could not be loaded. Checked: {checked_paths}")


def prepare_face(gray_frame, face_box):
    x, y, w, h = face_box
    roi_gray = gray_frame[y : y + h, x : x + w]
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_gray.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi


def predict_emotion(model, face_input):
    prediction = model.predict(face_input, verbose=0)[0]
    max_index = int(np.argmax(prediction))
    label = EMOTIONS.get(max_index, "unknown")
    confidence = float(np.max(prediction))
    probabilities = {
        EMOTIONS[index]: float(score) for index, score in enumerate(prediction)
    }
    return label, confidence, probabilities


def detect_emotions(frame_bgr, model, detector, scale_factor=1.1, min_neighbors=5):
    output = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(60, 60),
    )

    results = []
    for face_number, (x, y, w, h) in enumerate(faces, start=1):
        face_input = prepare_face(gray, (x, y, w, h))
        label, confidence, probabilities = predict_emotion(model, face_input)
        color = EMOTION_COLORS.get(label, (0, 180, 120))
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        text_y = max(y - 10, 24)
        cv2.putText(
            output,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "face": face_number,
                "emotion": label,
                "confidence": confidence,
                "box": (int(x), int(y), int(w), int(h)),
                "probabilities": probabilities,
            }
        )

    return output, results


def results_table(results):
    rows = []
    for result in results:
        row = {
            "Face": result["face"],
            "Emotion": result["emotion"],
            "Confidence": round(result["confidence"], 3),
        }
        row.update(
            {
                emotion.capitalize(): round(probability, 3)
                for emotion, probability in result["probabilities"].items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_emotion_model()
        self.detector = load_face_detector()

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        annotated, _ = detect_emotions(image, self.model, self.detector)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def render_image_upload(model, detector):
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_image is None:
        return

    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("This image could not be decoded.")
        return

    annotated, results = detect_emotions(image_bgr, model, detector)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

    if results:
        st.dataframe(results_table(results), use_container_width=True)
    else:
        st.info("No face was detected in this image.")


def render_video_upload(model, detector):
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=False,
    )

    if uploaded_video is None:
        return

    frame_step = st.slider("Frame sampling", min_value=1, max_value=30, value=10)
    max_frames = st.slider("Preview frames", min_value=5, max_value=100, value=30)

    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as temp_file:
        temp_file.write(uploaded_video.read())
        temp_path = temp_file.name

    capture = cv2.VideoCapture(temp_path)
    if not capture.isOpened():
        os.unlink(temp_path)
        st.error("This video could not be opened by OpenCV.")
        return

    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    summary_rows = []
    processed = 0
    frame_index = 0

    with st.spinner("Processing video preview..."):
        while capture.isOpened() and processed < max_frames:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_step == 0:
                annotated, results = detect_emotions(frame, model, detector)
                frame_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )
                for result in results:
                    summary_rows.append(
                        {
                            "Frame": frame_index,
                            "Face": result["face"],
                            "Emotion": result["emotion"],
                            "Confidence": round(result["confidence"], 3),
                        }
                    )
                processed += 1
                status_placeholder.caption(f"Processed {processed} preview frame(s).")

            frame_index += 1

    capture.release()
    os.unlink(temp_path)

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    else:
        st.info("No face was detected in the sampled video frames.")


def render_live_camera():
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="emotion-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=EmotionVideoProcessor,
        async_processing=True,
    )


def main():
    st.set_page_config(
        page_title="Emotion Recognition",
        page_icon=":movie_camera:",
        layout="wide",
    )

    st.title("Emotion Recognition")

    try:
        model = load_emotion_model()
        detector = load_face_detector()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    mode = st.selectbox("Mode", ["Live camera", "Image upload", "Video upload"])

    if mode == "Live camera":
        render_live_camera()
    elif mode == "Image upload":
        render_image_upload(model, detector)
    elif mode == "Video upload":
        render_video_upload(model, detector)


if __name__ == "__main__":
    main()
