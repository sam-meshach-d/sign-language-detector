
---

# 🤟 Sign Language Detector

A **Streamlit-based application** to collect gesture datasets, train LSTM models, and run **real-time sign language inference** using **MediaPipe** + **TensorFlow**.
This project enables you to:

1. **Realtime Simulation** – Run the live detector with a webcam, visualize keypoints, and get instant predictions.
2. **Create Dataset** – Capture gesture sequences and save synchronized `.npy` (keypoints) and `.mp4` (videos).
3. **Train Custom Model** – Train LSTM-based models on your dataset with one click, track metrics, and save artifacts.

---

## 📂 Project Structure

```
├── app.py              # Home page with navigation
├── dataset.py          # Dataset creation interface
├── inference.py        # Realtime inference dashboard
├── train.py            # Training interface for custom models
├── config/             # Settings and paths
├── modules/            # Shared UI and style helpers
├── static/             # UI images (realtime.png, dataset.png, train_model.png)
├── data/               # Saved keypoint sequences (.npy)
├── video/              # Saved gesture recordings (.mp4)
├── models/             # Trained models (model.h5, scaler.pkl, meta.json)
└── requirements.txt    # Python dependencies
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sam-meshach-d/sign-language-detector
cd sign-language-detector
```

### 2. Create a Virtual Environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:

* `streamlit`
* `opencv-python`
* `mediapipe`
* `tensorflow`
* `scikit-learn`
* `joblib`
* `numpy`

---

## 🚀 Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

This will open the **Sign Language Detector UI** in your browser.

---

## 🖥️ Features

### 1. Realtime Simulation

* Uses your **webcam** with MediaPipe to track body & hand landmarks.
* Loads a trained LSTM model and predicts gestures live.
* Configurable **confidence threshold** to stabilize predictions.

### 2. Create Dataset

* Record synchronized gesture samples (`.npy` + `.mp4`) per label.
* Built-in **delete/reset controls** for cleaning datasets.

### 3. Train Custom Model

* Trains an **LSTM classifier** with early stopping.
* Saves:

  * `model.h5` – trained TensorFlow model
  * `scaler.pkl` – MinMaxScaler for preprocessing
  * `meta.json` – metadata with labels, metrics, notes
* Manage models (rename, delete) directly from the UI.

---

## 📊 Data Storage

* **Datasets** → stored in `data/<gesture>/#####.npy`
* **Videos** → stored in `video/<gesture>/#####.mp4`
* **Models** → stored in `models/<run-name>/` containing:

  * `model.h5`
  * `scaler.pkl`
  * `meta.json`

---

## 🔑 Notes

* Ensure a **working webcam** is available.
* Good lighting and visible hands improve dataset quality.
* At least **2 gestures** are required to train a valid model.
* The **UI images** (`static/realtime.png`, `static/dataset.png`, `static/train_model.png`) must exist.

---