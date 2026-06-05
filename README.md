# Sign Language Recognition Using MobileNetV2

A real-time **American Sign Language (ASL)** recognition system powered by **MobileNetV2** and **MediaPipe Hands**. The application uses a webcam to detect hand gestures, classifies them into ASL letters, and lets users build words letter-by-letter through an interactive web interface.

---

## ✨ Features

- **Real-time webcam inference** — Hand region is detected per-frame and classified instantly.
- **MobileNetV2 backbone** — Lightweight CNN fine-tuned for 29-class ASL recognition (`A-Z`, `del`, `nothing`, `space`).
- **MediaPipe Hands integration** — Accurate hand landmark detection to crop the region of interest; falls back to center-crop when MediaPipe is unavailable.
- **Word builder UI** — Append predicted letters, insert spaces, and delete characters to construct sentences.
- **Modern glassmorphism interface** — Dark-themed, responsive web UI built with Flask + vanilla HTML/CSS/JS.

---

## 📁 Project Structure

```
sign-lang/
├── app.py                              # Flask server & inference pipeline
├── requirements.txt                    # Python dependencies
├── model/
│   ├── mobilenetv2_final_model.keras   # Trained MobileNetV2 model (~11 MB)
│   └── class_names.txt                 # 29 class labels (A-Z, del, nothing, space)
├── templates/
│   └── index.html                      # Web UI (glassmorphism dark theme)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.10+   |
| Webcam      | Any USB / built-in camera |
| OS          | Windows / macOS / Linux |

### Installation

```powershell
# 1. Clone the repository
git clone https://github.com/Abhiram4957/Sign-Language-Recognition.git
cd Sign-Language-Recognition

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the App

```powershell
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 🎮 How to Use

1. **Allow webcam access** when prompted by the browser.
2. Hold your hand inside the camera frame — the app detects and highlights it with a bounding box.
3. The **Current Prediction** panel shows the recognized ASL letter in real time.
4. Click **Append Letter** to add the predicted letter to the word builder.
5. Use **Reset Word** to clear the built word.

### Special Classes

| Prediction | Action |
|------------|--------|
| `A` – `Z`  | Appends the letter to the word |
| `space`    | Inserts a blank space |
| `del`      | Removes the last character |
| `nothing`  | Ignored (no hand detected) |

---

## 🧠 Model Details

| Property         | Value |
|------------------|-------|
| Architecture     | MobileNetV2 (transfer learning) |
| Input size       | 128 × 128 × 3 |
| Output classes   | 29 (A-Z + del, nothing, space) |
| Framework        | TensorFlow / Keras |
| Format           | `.keras` |

The model was fine-tuned on an ASL alphabet dataset. Hand regions are preprocessed with Gaussian blur, resized to 128×128, and normalized to `[0, 1]` before inference.

---

## 🔧 Tech Stack

- **Backend**: Flask
- **Deep Learning**: TensorFlow / Keras (MobileNetV2)
- **Hand Detection**: MediaPipe Hands
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS (glassmorphism), vanilla JavaScript

---

## 📦 Dependencies

```
flask>=3.0
mediapipe>=0.10
numpy>=1.26
opencv-python>=4.8
tensorflow>=2.16
```

---

## 📝 Notes

- The trained model file (`mobilenetv2_final_model.keras`) is included in the `model/` directory.
- The runtime label order is stored in `model/class_names.txt`. If your model was trained with a different class order, update this file to match.
- If MediaPipe Hands is not available in your environment, the app automatically falls back to a center-crop strategy.

---

## 📄 License

This project is open source and available for educational and research purposes.

---

## 🙏 Acknowledgements

- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al.
- [MediaPipe](https://mediapipe.dev/) — Google
- ASL Alphabet Dataset — Community datasets for American Sign Language recognition
