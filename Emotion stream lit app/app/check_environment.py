from pathlib import Path


def check_import(name, label=None):
    module = __import__(name)
    version = getattr(module, "__version__", "installed")
    print(f"{label or name}: {version}")
    return module


def main():
    print("Checking Emotion Recognition app environment...")
    print(f"Project folder: {Path(__file__).resolve().parent}")
    print()

    check_import("streamlit")
    check_import("tensorflow")
    check_import("cv2", "opencv")
    check_import("av")
    check_import("streamlit_webrtc")
    check_import("pandas")
    check_import("numpy")

    from app import MODEL_JSON_PATH, MODEL_WEIGHTS_PATH, load_face_detector, load_emotion_model

    print()
    print(f"Model JSON exists: {MODEL_JSON_PATH.exists()} - {MODEL_JSON_PATH}")
    print(f"Model weights exist: {MODEL_WEIGHTS_PATH.exists()} - {MODEL_WEIGHTS_PATH}")

    model = load_emotion_model()
    detector = load_face_detector()
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Face detector loaded: {not detector.empty()}")
    print()
    print("Environment check passed.")


if __name__ == "__main__":
    main()
