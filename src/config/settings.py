from pathlib import Path
# =========================
# Project-wide Settings
# =========================
PROJECT_NAME = "Sign-Language-Detector"
RANDOM_SEED = 42

# =========================
# Data Collection Settings
# =========================
SEQ_LEN = 30               # frames per sequence
FPS = 30                   # frames per second
RESOLUTION = (640, 480)    # camera resolution
LANDMARKS_LEN = 225
DEFAULT_CAMERA = 0
# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = "data"
VIDEO_PATH = "video"
MODELS_PATH = "models"

CSS_FILE = "styles.css"

INFERENCE_FILE = "pages/inference.py"
DATASET_FILE = "pages/dataset.py"
TRAIN_FILE = "pages/train.py"
HOME_FILE = "app.py"
# =========================
# MediaPipe Settings
# =========================
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# =========================
# Training Settings
# =========================
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
