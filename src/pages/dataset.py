import streamlit as st
import cv2
import os
import numpy as np
import mediapipe as mp
import shutil
from datetime import datetime
from modules.sidebar import render_sidebar
from modules.style import load_css
from config.settings import (
    SEQ_LEN, FPS, RESOLUTION,
    DATA_PATH, VIDEO_PATH,
    MIN_DETECTION_CONF, MIN_TRACKING_CONF, CSS_FILE, DEFAULT_CAMERA
)

# ======= MediaPipe Setup =======
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =============== THEME & STYLES (match Home) ===============
st.set_page_config(page_title="Create Dataset • Sign Language", page_icon="📸", layout="wide")
render_sidebar()
load_css(CSS_FILE)


# =============== HELPERS (logic preserved) ===============
def list_all_gestures(data_path=DATA_PATH):
    gestures_info = []
    if os.path.exists(data_path):
        for gesture in sorted(os.listdir(data_path)):
            gesture_path = os.path.join(data_path, gesture)
            if os.path.isdir(gesture_path):
                count = len([f for f in os.listdir(gesture_path) if f.endswith(".npy")])
                gestures_info.append({"Gesture": gesture, "Samples": count})
    return gestures_info

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, lh, rh], axis=0).flatten()

def get_existing_samples(gesture_name):
    gesture_path = os.path.join(DATA_PATH, gesture_name)
    if not os.path.exists(gesture_path):
        return []
    return [f for f in os.listdir(gesture_path) if f.endswith(".npy")]

def collect_samples(gesture_name, seq_len, total_samples, resolution, fps, camera_index):
    DATA_NP_PATH = os.path.join(DATA_PATH, gesture_name)
    DATA_VIDEO_PATH = os.path.join(VIDEO_PATH, gesture_name)
    os.makedirs(DATA_NP_PATH, exist_ok=True)
    os.makedirs(DATA_VIDEO_PATH, exist_ok=True)

    existing_files = [f for f in os.listdir(DATA_NP_PATH) if f.endswith('.npy')]
    existing_ids = [int(f.replace(".npy", "")) for f in existing_files if f.replace(".npy", "").isdigit()]
    st.session_state.sample_counter = max(existing_ids) if existing_ids else 0

    if "stop_flag" not in st.session_state:
        st.session_state.stop_flag = False
    if "reset_flag" not in st.session_state:
        st.session_state.reset_flag = False

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    holistic = mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONF, min_tracking_confidence=MIN_TRACKING_CONF)

    cam_col = st.columns(3)[1]
    with cam_col:
        frame_placeholder = st.empty()
        status_text = st.empty()
        sample_info = st.empty()
        progress = st.progress(0)
        progress.progress(st.session_state.sample_counter / total_samples)

    while st.session_state.sample_counter < total_samples:
        sequence, keypoints_sequence = [], []
        sample_info.markdown(f"#### 📸 Capturing Sample {st.session_state.sample_counter + 1} / {total_samples}")

        while len(sequence) < seq_len:
            if st.session_state.stop_flag:
                status_text.warning("⏹️ Stopped by user.")
                cap.release()
                holistic.close()
                return

            ret, frame = cap.read()
            if not ret:
                status_text.error("❌ Failed to read frame from camera.")
                continue

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if results.pose_landmarks and (results.left_hand_landmarks or results.right_hand_landmarks):
                keypoints = extract_keypoints(results)
                sequence.append(frame)
                keypoints_sequence.append(keypoints)
                cv2.putText(image, f"Frame {len(sequence)}/{seq_len}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 120), 2)
            else:
                cv2.putText(image, "❌ No landmarks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 120, 120), 2)

            frame_placeholder.image(image, channels="RGB")

            if st.session_state.reset_flag:
                status_text.info("🔁 Resetting current sample...")
                sequence, keypoints_sequence = [], []
                st.session_state.reset_flag = False

        file_id = f"{st.session_state.sample_counter+1:05d}"
        np.save(os.path.join(DATA_NP_PATH, f"{file_id}.npy"), np.array(keypoints_sequence))

        video_path = os.path.join(DATA_VIDEO_PATH, f"{file_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, resolution)
        for f in sequence:
            out.write(f)
        out.release()

        st.session_state.sample_counter += 1
        progress.progress(st.session_state.sample_counter / total_samples)
        status_text.success(f"✅ Sample {file_id} saved.")

    cap.release()
    holistic.close()
    status_text.write("🎉 Data collection complete!")
    st.session_state.recording = False
    st.rerun()

# =============== SESSION STATE ===============
if "recording" not in st.session_state:
    st.session_state.recording = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "reset_flag" not in st.session_state:
    st.session_state.reset_flag = False

# =============== HEADER ===============
st.markdown(f"""
<div class="hero">
  <span class="badge">📸 Gestures • Landmarks • Custom</span>
  <h2>Create Dataset</h2>
  <p class="subtle">Capture synchronized keypoints and video clips per gesture to train a robust sign language model.</p>
  <div>
    <span class="pill">SEQUENCE LENGTH: {SEQ_LEN}</span>
    <span class="pill">FPS: {FPS}</span>
    <span class="pill">RESOLUTION: {RESOLUTION[0]}×{RESOLUTION[1]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Dataset Overview")
# Compute your values in Python
gestures = list_all_gestures(DATA_PATH)
total_gestures = len(gestures)
total_samples = sum(g["Samples"] for g in gestures) if gestures else 0

st.markdown(f"""
<div class="kpi-wrap">
  <div class="kpi-bar">
    <div class="kpi-tile">
      <div class="kpi-label">Total Gestures</div>
      <div class="kpi-value">{total_gestures}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Total Samples</div>
      <div class="kpi-value">{total_samples}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">Sequence Length</div>
      <div class="kpi-value">{SEQ_LEN}</div>
    </div>
    <div class="kpi-tile">
      <div class="kpi-label">FPS</div>
      <div class="kpi-value">{FPS}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Dataset list inside an expander
with st.expander("View gestures and sample counts", expanded=False):
    if gestures:
        gestures_sorted = sorted(gestures, key=lambda x: x["Gesture"])
        rows = "\n".join([f"<tr><td>{g['Gesture']}</td><td>{g['Samples']}</td></tr>" for g in gestures_sorted])
        st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="tbl">
          <thead><tr><th>Gesture</th><th>Samples</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No gestures collected yet. Create your first entry below.")

st.caption(f"Updated {datetime.now().strftime('%b %d, %Y %H:%M')}")

st.markdown("""
<div class="learn-wrap">
  <div class="learn-head">
    <span class="use-badge">🤟 Learn Sign Language</span>
    <div class="learn-title">Build better datasets with accurate signs</div>
    <div class="learn-sub">Quick courses and references to improve labeling quality and recording consistency.</div>
  </div>

  <div class="learn-grid">
    <!-- ISL -->
    <div class="learn-card">
      <div class="learn-card-head">
        <div class="learn-card-title">Indian Sign Language (ISL)</div>
        <div class="learn-card-sub">Official self-learning course from ISLRTC</div>
      </div>
      <ul class="learn-points">
        <li>Basics to situation-based conversations</li>
        <li>Vocabulary, grammar, etiquette</li>
        <li>Module quizzes for self‑assessment</li>
      </ul>
      <a class="learn-cta" href="https://islrtc.nic.in/online-basic-isl-course-in-self-learning-mode/" target="_blank" rel="noopener">
        📚 Start ISL Course
      </a>
    </div>
    <!-- ASL -->
    <div class="learn-card">
      <div class="learn-card-head">
        <div class="learn-card-title">American Sign Language (ASL)</div>
        <div class="learn-card-sub">Lifeprint: lessons, videos, and fingerspelling</div>
      </div>
      <ul class="learn-points">
        <li>Structured lessons and vocab indexes</li>
        <li>Handshape, numbers, fingerspelling</li>
        <li>Great for reference during labeling</li>
      </ul>
      <a class="learn-cta" href="https://www.lifeprint.com/" target="_blank" rel="noopener">
        📘 Explore ASL (Lifeprint)
      </a>
    </div>
  </div>

  <div class="learn-note">
    Tip: Standardize labels and capture protocols after completing a basics module for consistency across contributors.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="dc-panel">
  <div class="dc-head">
    <span class="dc-badge">Guide</span>
    Efficient data collection — clean sequences, consistent samples
  </div>
  <p class="dc-sub">Capture high‑quality examples for robust training. Follow this rhythm so each saved sample contains exactly {SEQ_LEN} frames of a single gesture with clear landmarks.</p>

  <div class="dc-grid">
    <div class="dc-card">
      <div class="t">1) Prepare the frame</div>
      <div class="v">Set the correct camera, ensure good lighting, and keep background motion minimal. Keep both hands fully visible when your gesture needs them.</div>
    </div>
    <div class="dc-card">
      <div class="t">2) Fill the sequence</div>
      <div class="v">Hold the gesture steadily until the counter reaches <strong>Frame {SEQ_LEN}/{SEQ_LEN}</strong>. Only frames with valid pose + at least one hand are recorded; drifting out of frame will pause filling.</div>
    </div>
    <div class="dc-card">
      <div class="t">3) Save per‑sample artifacts</div>
      <div class="v">After collecting {SEQ_LEN} frames, the app saves synchronized files: <code>data/&lt;gesture&gt;/#####.npy</code> and <code>video/&lt;gesture&gt;/#####.mp4</code>, with incremental IDs.</div>
    </div>
    <div class="dc-card">
      <div class="t">4) Use controls effectively</div>
      <div class="v">Use <strong>Reset</strong> to clear the current sequence if you made a mistake. Use <strong>Stop</strong> for a clean exit; the app closes the camera and releases resources safely.</div>
    </div>
  </div>

  <div class="dc-flow">
    <span class="dc-chip"><strong>Pose hands</strong> in frame</span>
    <span class="dc-chip"><strong>Hold</strong> until Frame {SEQ_LEN}/{SEQ_LEN}</span>
    <span class="dc-chip"><strong>Save</strong> auto‑writes .npy + .mp4</span>
    <span class="dc-chip"><strong>Hands off</strong> to segment</span>
    <span class="dc-chip"><strong>Repeat</strong> next sample</span>
  </div>

  <div class="dc-mini">
    <div class="dc-pair">Landmarks required: <code>pose</code> + <code>left or right hand</code></div>
    <div class="dc-pair">Per‑gesture folders: <code>data/&lt;gesture&gt;</code>, <code>video/&lt;gesture&gt;</code></div>
    <div class="dc-pair">Auto ID: continues from last saved file (e.g., 00007.npy)</div>
  </div>

  <div class="dc-note">
    Tip: If “❌ No landmarks” appears, pause and re‑center. Keep wrists and palms visible. Use consistent distance/orientation for comparable samples.
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("### ⚙️ Capture Settings")

col1, col2, col3 = st.columns([1.4, 1, 1])
with col1:
    gesture_name = st.text_input("Gesture Name", "abc")
with col2:
    target_samples = st.number_input("Target Samples", min_value=1, max_value=1000, value=50)
with col3:
    camera_index = st.selectbox("Select Camera", options=[0, 1, 2], index=DEFAULT_CAMERA)

existing_samples = get_existing_samples(gesture_name)
sample_count = len(existing_samples)
st.write(f"📂 Existing for {gesture_name}: {sample_count}/{target_samples}")

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("▶️ Start Recording", key="start_btn", use_container_width=True):
        st.session_state.recording = True
        st.session_state.stop_flag = False
        st.session_state.reset_flag = False
with b2:
    if st.button("⏹️ Stop", key="stop_btn", use_container_width=True):
        st.session_state.stop_flag = True
        st.session_state.recording = False
with b3:
    if st.button("🔁 Reset Sample", key="reset_btn", use_container_width=True):
        st.session_state.reset_flag = True

st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

d1, d2 = st.columns(2)
with d1:
    if st.button("🗑️ Delete All Samples", key="delete_all_btn", use_container_width=True):
        gesture_np_path = os.path.join(DATA_PATH, gesture_name)
        gesture_video_path = os.path.join(VIDEO_PATH, gesture_name)
        deleted = False
        if os.path.exists(gesture_np_path):
            shutil.rmtree(gesture_np_path)
            deleted = True
        if os.path.exists(gesture_video_path):
            shutil.rmtree(gesture_video_path)
            deleted = True
        st.session_state.delete_message = f"All samples (npy + video) for {gesture_name} have been deleted!" if deleted else "No samples found to delete."
        st.session_state.recording = False
        st.rerun()
with d2:
    if st.button("🗑️ Delete Latest Sample", key="delete_latest_btn", use_container_width=True):
        gesture_np_path = os.path.join(DATA_PATH, gesture_name)
        gesture_video_path = os.path.join(VIDEO_PATH, gesture_name)
        if os.path.exists(gesture_np_path):
            files = sorted([f for f in os.listdir(gesture_np_path) if f.endswith(".npy")])
            if files:
                latest_id = files[-1].replace(".npy", "")
                os.remove(os.path.join(gesture_np_path, files[-1]))
                video_file = os.path.join(gesture_video_path, f"{latest_id}.mp4")
                if os.path.exists(video_file):
                    os.remove(video_file)
                st.session_state.delete_message = f"Latest sample {latest_id} (npy + video) deleted for {gesture_name}!"
            else:
                st.session_state.delete_message = "No sample files found to delete."
        else:
            st.session_state.delete_message = "No samples found for this gesture."
        st.rerun()

if "delete_message" in st.session_state:
    st.toast(st.session_state.delete_message, icon="ℹ️")
    del st.session_state.delete_message

st.markdown('</div>', unsafe_allow_html=True)

# =============== CAMERA CAPTURE ===============
if st.session_state.recording:
    collect_samples(gesture_name, SEQ_LEN, target_samples, RESOLUTION, FPS, camera_index)
else:
    if sample_count >= target_samples and target_samples > 0:
        st.success(f"✅ Dataset for {gesture_name} is full!")
