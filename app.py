from __future__ import annotations

import atexit
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "mobilenetv2_final_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.txt"
DEFAULT_CLASSES = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + ["del", "nothing", "space"]

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

MP_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions")
mp_hands = mp.solutions.hands if MP_SOLUTIONS_AVAILABLE else None
mp_draw = mp.solutions.drawing_utils if MP_SOLUTIONS_AVAILABLE else None
hands = None
camera = None
word = ""
current_letter = ""


def load_classes(expected_count: int) -> list[str]:
    if CLASS_NAMES_PATH.exists():
        classes = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(classes) == expected_count:
            return classes

    if len(DEFAULT_CLASSES) == expected_count:
        return DEFAULT_CLASSES.copy()

    return [f"class_{index}" for index in range(expected_count)]


model = load_model(MODEL_PATH)
CLASSES = load_classes(int(model.output_shape[-1]))


def get_hands():
    global hands

    if not MP_SOLUTIONS_AVAILABLE:
        return None

    if hands is None:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
        )

    return hands


def get_camera():
    global camera

    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    return camera


def release_camera():
    global camera

    if camera is not None and camera.isOpened():
        camera.release()
    camera = None


atexit.register(release_camera)


def encode_status_frame(message: str) -> bytes:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera unavailable", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 180, 255), 2)
    cv2.putText(frame, message, (80, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


def get_center_crop(frame: np.ndarray) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    height, width, _ = frame.shape
    crop_size = int(min(height, width) * 0.65)
    half_size = crop_size // 2
    center_x, center_y = width // 2, height // 2
    x_min = max(center_x - half_size, 0)
    y_min = max(center_y - half_size, 0)
    x_max = min(x_min + crop_size, width)
    y_max = min(y_min + crop_size, height)
    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None, None

    return roi, (x_min, y_min, x_max, y_max)


def detect_hand_roi(
    frame: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, object | None]:
    height, width, _ = frame.shape
    detector = get_hands()

    if detector is None:
        roi, bounds = get_center_crop(frame)
        return roi, bounds, None

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb_frame)

    if not result.multi_hand_landmarks:
        return None, None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    x_min, y_min = width, height
    x_max, y_max = 0, 0

    for landmark in hand_landmarks.landmark:
        x_pos, y_pos = int(landmark.x * width), int(landmark.y * height)
        x_min, y_min = min(x_pos, x_min), min(y_pos, y_min)
        x_max, y_max = max(x_pos, x_max), max(y_pos, y_max)

    margin = 30
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, width)
    y_max = min(y_max + margin, height)
    roi = frame[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        return None, None, None

    return roi, (x_min, y_min, x_max, y_max), hand_landmarks


def gen_frames():
    global current_letter, word

    while True:
        cap = get_camera()
        if cap is None or not cap.isOpened():
            frame = encode_status_frame("Check webcam permissions and reconnect the camera.")
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.25)
            continue

        success, frame = cap.read()
        if not success:
            frame = encode_status_frame("Unable to read a frame from the webcam.")
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        detected_letter = ""
        hand_roi, bounds, hand_landmarks = detect_hand_roi(frame)

        if hand_roi is not None:
            x_min, y_min, x_max, y_max = bounds
            hand_roi = cv2.GaussianBlur(hand_roi, (5, 5), 0)
            hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
            hand_roi = cv2.resize(hand_roi, (128, 128)).astype("float32") / 255.0
            hand_roi = np.expand_dims(hand_roi, axis=0)

            predictions = model.predict(hand_roi, verbose=0)
            predicted_index = int(np.argmax(predictions))
            detected_letter = CLASSES[predicted_index]
            current_letter = detected_letter

            box_color = (0, 255, 0) if MP_SOLUTIONS_AVAILABLE else (0, 180, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(
                frame,
                detected_letter,
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                box_color,
                2,
            )

            if hand_landmarks is not None and mp_draw is not None and mp_hands is not None:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if not MP_SOLUTIONS_AVAILABLE:
            cv2.putText(
                frame,
                "MediaPipe Hands not available: using center crop fallback",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                2,
            )

        if not detected_letter:
            current_letter = ""

        cv2.putText(frame, f"Word: {word}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        _, buffer = cv2.imencode(".jpg", frame)
        encoded_frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + encoded_frame + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_word")
def get_word():
    return jsonify({"word": word, "current_letter": current_letter})


@app.route("/reset_word")
def reset_word():
    global word
    word = ""
    return jsonify({"status": "reset", "word": word})


@app.route("/append_letter")
def append_letter():
    global current_letter, word

    if current_letter:
        lowered = current_letter.lower()
        if lowered == "space":
            word += " "
        elif lowered == "del":
            word = word[:-1]
        elif lowered != "nothing":
            word += current_letter

    return jsonify({"word": word, "current_letter": current_letter})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
