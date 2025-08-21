import streamlit as st
from config.settings import ROOT_DIR, CSS_FILE, INFERENCE_FILE, DATASET_FILE, TRAIN_FILE
from modules.sidebar import render_sidebar
from modules.style import load_css
import base64
from pathlib import Path
st.set_page_config(
    page_title="Sign Language Detector",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css(CSS_FILE)
render_sidebar()

def to_data_uri(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else ("image/jpeg" if ext in ("jpg", "jpeg") else "image/webp")
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

static = ROOT_DIR / "static"

realtime_img   = static / "realtime.png"
dataset_img    = static / "dataset.png"
training_img   = static / "train_model.png"

for p in (realtime_img, dataset_img, training_img):
    if not p.exists():
        raise FileNotFoundError(f"Missing image: {p}")
    
realtime_uri = to_data_uri(realtime_img)
dataset_uri  = to_data_uri(dataset_img)
training_uri = to_data_uri(training_img)

st.markdown(f"""
<div class="hero">
  <span class="badge">🤖 OpenCV • MediaPipe • Streamlit</span>
  <h1>Sign Language Detector</h1>
  <p>Collect high-quality sequences, train a custom model, and validate predictions in realtime—optimized for rapid iteration.</p>
  <div class="cards">
    <!-- Card 1 -->
    <div class="card flip-card">
      <div class="flip-inner">
        <!-- FRONT -->
        <div class="flip-front">
          <h3>Realtime Simulation</h3>
          <p class="clamp-2">Run the live detector on webcam and visualize keypoints and predictions instantly.</p>
          <div class="footer-row">
            <div class="kpis">
              <div class="kpi">Webcam</div>
              <div class="kpi">MediaPipe</div>
              <div class="kpi">Live FPS</div>
            </div>
          </div>
        </div>
        <!-- BACK -->
        <div class="flip-back">
          <img src={realtime_uri} alt="Realtime Simulation Preview"/>
          <div class="flip-overlay">
          </div>
        </div>
      </div>
    </div>
    <!-- Card 2 -->
    <div class="card flip-card">
      <div class="flip-inner">
        <div class="flip-front">
          <h3>Create Dataset</h3>
          <p class="clamp-2">Capture sequences and save synchronized .npy keypoints and videos for robust training.</p>
          <div class="footer-row">
            <div class="kpis">
              <div class="kpi">Sequences</div>
              <div class="kpi">Labels</div>
              <div class="kpi">Versioned</div>
            </div>
          </div>
        </div>
        <div class="flip-back">
          <img src={dataset_uri} alt="Dataset Preview"/>
          <div class="flip-overlay">
          </div>
        </div>
      </div>
    </div>
    <!-- Card 3 -->
    <div class="card flip-card">
      <div class="flip-inner">
        <div class="flip-front">
          <h3>Train Custom Model</h3>
            <p class="clamp-2">One-click training with smart defaults—tracks metrics and exports the best checkpoint automatically.</p>          <div class="footer-row">
            <div class="kpis">
              <div class="kpi">LSTM</div>
              <div class="kpi">TensorFlow</div>
              <div class="kpi">ONNX Export</div>
            </div>
          </div>
        </div>
        <div class="flip-back">
          <img src={training_uri} alt="Training Preview"/>
          <div class="flip-overlay">
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    wrap = st.container()
    with wrap:
        if st.button("▶ Realtime Simulation", key="cta_rt_below", use_container_width=True):
            st.switch_page(INFERENCE_FILE)

with c2:
    wrap = st.container()
    with wrap:
        if st.button("📸 Create Dataset", key="cta_ds_below", use_container_width=True):
            st.switch_page(DATASET_FILE)

with c3:
    wrap = st.container()
    with wrap:
        if st.button("🧠 Train Custom Model", key="cta_tr_below", use_container_width=True):
            st.switch_page(TRAIN_FILE)
