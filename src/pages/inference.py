import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from modules.sidebar import render_sidebar
import json
import joblib
from tensorflow.keras.models import load_model
from modules.style import load_css
from collections import deque
from config.settings import (
    SEQ_LEN, FPS, RESOLUTION, LANDMARKS_LEN, MODELS_PATH,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF, CSS_FILE, DEFAULT_CAMERA
)



MODELS_DIR = Path(MODELS_PATH)
# =============== THEME & STYLES (match Home) ===============
st.set_page_config(page_title="Realtime Simulation • Sign Language", page_icon="🎥", layout="wide")
render_sidebar()
load_css(CSS_FILE)

# =============== HEADER ===============
st.markdown(f"""
<div class="hero">
  <span class="badge">🎥 Realtime • LSTM • MediaPipe</span>
  <h2>Inference Dashboard</h2>
  <p class="subtle">Run live sign language predictions with your trained sequence models, see smoothed results, and confidence in real‑time.</p>
  <div>
    <span class="pill">SEQUENCE LENGTH: {SEQ_LEN}</span>
    <span class="pill">FPS: {FPS}</span>
    <span class="pill">RESOLUTION: {RESOLUTION[0]}×{RESOLUTION[1]}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def list_runs(models_dir: Path):
    runs = []
    if models_dir.exists():
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "model.h5").exists() and (p / "scaler.pkl").exists():
                runs.append(p)
    return runs

def load_meta(run_dir: Path):
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, lh, rh], axis=0).flatten()  # (225,)

@st.cache_resource
def load_artifacts(run_dir):
    with st.spinner("🔄 Loading model and scaler..."):
        model = load_model(run_dir / "model.h5")
        scaler = joblib.load(run_dir / "scaler.pkl")
    return model, scaler

# ---------- Discover runs ----------
runs = list_runs(MODELS_DIR)
if not runs:
    st.error("No model found. Please train one to continue.")
    st.stop()

run_names = [r.name for r in runs]
default_idx = max(0, len(run_names) - 1)
st.subheader("Choose a model")
chosen_run_name = st.selectbox("Trained models", run_names, index=default_idx)
chosen_run = MODELS_DIR / chosen_run_name
  
model, scaler = load_artifacts(chosen_run)
meta = load_meta(chosen_run)

rev_label_map = {int(k) : v for k, v in meta['labels'].items()}

st.markdown("### Model Overview")


st.markdown(f"""
<div class="kpi-wrap">
  <div class="kpi-bar">
    <div class="kpi-tile">
      <div class="kpi-label">Model name</div>
      <div class="kpi-value">{meta['model_name']}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Created</div>
      <div class="kpi-value">{meta['created_at']}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Classes</div>
      <div class="kpi-value">{len(meta['label_map'])}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Notes</div>
      <div class="kpi-value">{meta['notes']}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
def fmt_pct(x, default="—"):
    try:
        if x is None or x == "—":
            return default
        if isinstance(x, str):
            x = x.strip().replace("%", "")
            x = float(x) if float(x) > 1 else float(x) * 100
            return f"{x:.0f}%"
        if isinstance(x, (int, float)):
            x = x * 100 if x <= 1 else x
            return f"{x:.0f}%" if abs(x - round(x)) < 1e-6 else f"{x:.1f}%"
        return default
    except Exception:
        return default
    
st.subheader("Metrics")
m = meta.get("metrics", {})

metrics_to_show = []
if m:
    for k in ["test_accuracy", "precision_macro", "recall_macro", "f1_macro",
              "precision_weighted", "recall_weighted", "f1_weighted"]:
        if k in m:
            metrics_to_show.append((k.replace("_", " ").title(), fmt_pct(m[k])))
else:
    st.info("No metrics in meta.json")

# Now render KPI tiles
if metrics_to_show:
    tiles_html = '<div class="kpi-wrap"><div class="kpi-bar">'
    for label, value in metrics_to_show:
        tiles_html += f"""
        <div class="kpi-tile">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>"""
    tiles_html += "</div></div>"

    st.markdown(tiles_html, unsafe_allow_html=True)


st.markdown('<div></div>', unsafe_allow_html=True)
labels = meta.get("labels", {})

with st.expander("Gestures the model can detect", expanded=False):
    if labels:
        try:
            ordered_items = sorted(labels.items(), key=lambda kv: int(kv[0]))
        except Exception:
            ordered_items = sorted(labels.items(), key=lambda kv: kv)
        rows_html = "\n".join(
            [f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in ordered_items]
        )
        st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <table class="tbl">
            <thead>
                <tr><th>Index</th><th>Gesture</th></tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No labels.")

st.markdown('</div>', unsafe_allow_html=True)
# Usage guide panel (place near top of the page, after load_css)
st.markdown(f"""
<div class="use-panel">
  <div class="use-head">
    <span class="use-badge">Guide</span>
    Live Inference — How to get clean, stable predictions
  </div>
  <p class="use-sub">Use this rhythm to minimize spillover and boost confidence.</p>

  <div class="use-steps">
    <div class="use-step">
      <div class="t">1. Prepare</div>
      <div class="v">Good lighting, correct camera index, steady frame. Keep hands visible.</div>
    </div>
    <div class="use-step">
      <div class="t">2. Perform</div>
      <div class="v">Hold the gesture steadily for a full window ({SEQ_LEN} frames). Avoid drifting out of frame.</div>
    </div>
    <div class="use-step">
      <div class="t">3. Confirm</div>
      <div class="v">Watch the label and confidence. Green text (Confidence threshold) indicates stable prediction.</div>
    </div>
    <div class="use-step">
      <div class="t">4. Clear</div>
      <div class="v">Briefly take hands out of frame. This triggers “no hands” and resets the buffer.</div>
    </div>
    <div class="use-step">
      <div class="t">5. Repeat</div>
      <div class="v">Re-enter and perform the next gesture (or retry). Each segment starts clean.</div>
    </div>
  </div>

  <div class="use-flow">
    <span class="use-chip"><strong>Perform</strong></span>
    <span class="use-chip"><strong>Confirm</strong> label + confidence</span>
    <span class="use-chip"><strong>Clear</strong> hands out of frame</span>
    <span class="use-chip"><strong>Perform</strong> next</span>
  </div>

  <div class="use-note">
    Tip: The app predicts only after {SEQ_LEN} valid frames. Any “no hands” moment clears the sequence and prevents previous gesture spillover.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    confidence_gate = float(c1.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05))
with c2:
    camera_index = st.selectbox("Select Camera", options=[0, 1, 2], index=DEFAULT_CAMERA)

start = st.toggle("Start/Stop realtime detection", value=False)

sequence = deque(maxlen=SEQ_LEN)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONF, min_tracking_confidence=MIN_TRACKING_CONF)

_, cam_col, _ = st.columns([1, 2, 1])
with cam_col:
  frame_holder = st.empty()
  if start:
      st.session_state["running"] = True
      cap = cv2.VideoCapture(int(camera_index))
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

      try:
        while True:
              if not st.session_state.get("running", False):
                  break  # clean exit when toggle flips off
              ret, frame = cap.read()
              if not ret:
                  break

              img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              img.flags.writeable = False
              results = holistic.process(img)
              img.flags.writeable = True
              img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
              mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
              mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
              mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

              if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
                  keypoints = extract_keypoints(results)
                  sequence.append(keypoints)

                  if len(sequence) == SEQ_LEN:
                      input_np = np.array(sequence)                     
                      input_flat = input_np.reshape(1, -1)              
                      input_scaled = scaler.transform(input_flat)       
                      input_data = input_scaled.reshape(1, SEQ_LEN, LANDMARKS_LEN)
                      probs = model.predict(input_data)[0]
                      pred_idx = np.argmax(probs)
                      pred_label = rev_label_map[pred_idx]
                      confidence = probs[pred_idx]

                      cv2.putText(img, f'{pred_label} ({confidence:.2f})', (20, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if confidence > 0.7 else (0, 0, 255), 2)

              else:
                  sequence.clear()
                  cv2.putText(img, "❌ No hands detected", (20, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
              frame_holder.image(img, channels="BGR")
      finally:
          holistic.close()
          cap.release()
  else:
      st.session_state["running"] = False
      st.stop()

