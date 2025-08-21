import json, time, shutil, re
from pathlib import Path
import numpy as np
import streamlit as st
import joblib
from config.settings import DATA_PATH, MODELS_PATH, CSS_FILE
from modules.sidebar import render_sidebar
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from modules.style import load_css

DATA_DIR = Path(DATA_PATH)
MODELS_DIR = Path(MODELS_PATH)
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

st.set_page_config(page_title="Train Custom Model", page_icon="🧠", layout="wide")
render_sidebar()
load_css(CSS_FILE)

st.session_state.setdefault("op_status", None)
st.session_state.setdefault("last_model_sel", None)

if st.session_state.get("op_status"):
    kind = st.session_state["op_status"]["type"]
    msg  = st.session_state["op_status"]["msg"]
    # Map toasts
    if kind == "success":
        st.toast(msg, icon="✅")
    elif kind == "warning":
        st.toast(msg, icon="⚠️")
    elif kind == "error":
        st.toast(msg, icon="❌")
    else:
        st.toast(msg, icon="ℹ️")
    st.session_state["op_status"] = None


# =============== HEADER ===============
st.markdown("""
<div class="hero">
  <span class="badge">🤖 LSTM • Accelerated Training • Production-ready</span>
  <h1>🧠 Train a Custom Sign Language Model</h1>
  <p class="subtle">
    Ingest curated sequences, fine‑tune an LSTM, and export a high‑accuracy classifier—built for rapid iteration and real‑time inference.
  </p>
  <ul class="hero-highlights">
    <li>One‑click training with smart validation and test metrics</li>
    <li>Consistent preprocessing with saved scaler for seamless deployment</li>
  </ul>
</div>
""", unsafe_allow_html=True)


# ---------- Helpers ----------
def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9-_]+", "-", name.strip()).strip("-").lower()
    return re.sub(r"-{2,}", "-", s) or "model"

def unique_model_dir(base_name: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    folder = MODELS_DIR / f"{slugify(base_name)}-{ts}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder

def list_gestures(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

def load_dataset(gestures: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    X, y = [], []
    label_map = {g: i for i, g in enumerate(gestures)}
    for g in gestures:
        gdir = DATA_DIR / g
        for f in sorted(gdir.glob("*.npy")):
            seq = np.load(f)
            if np.count_nonzero(seq) > 0:
                X.append(seq)
                y.append(label_map[g])
    if not X:
        return np.array([]), np.array([]), label_map
    X = np.array(X)
    y = np.array(y)
    return X, y, label_map

def train_model(X, y, seed=SEED, test_size=TEST_SIZE, val_size=VAL_SIZE):
    n_samples, T, F = X.shape
    X_flat = X.reshape(n_samples, -1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(n_samples, T, F)

    y_cat = to_categorical(y)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_cat, test_size=test_size, random_state=seed, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio,
        random_state=seed, stratify=np.argmax(y_train_val, axis=1)
    )

    input_shape = X_train.shape[1:]
    num_classes = y_cat.shape[1]

    model = Sequential([
        LSTM(32, activation="relu", input_shape=input_shape),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=100, batch_size=16, callbacks=[early], verbose=0
    )

    val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precision_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics = {
        "train_size": X_train.shape,
        "val_size": X_val.shape,
        "test_size": X_test.shape,
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
    }
    return model, scaler, metrics

def save_artifacts(folder: Path, model, scaler, meta: dict):
    model.save(folder / "model.h5")
    joblib.dump(scaler, folder / "scaler.pkl")
    with open(folder / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_models_index() -> list[Path]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p for p in MODELS_DIR.iterdir() if p.is_dir()])

def read_meta(folder: Path) -> dict:
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return {"model_name": folder.name, "created_at": None}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {"model_name": folder.name, "created_at": None}

def rename_model_folder(old_folder: Path, new_base: str) -> Path:
    new_name = f"{slugify(new_base)}-{time.strftime('%Y%m%d-%H%M%S')}"
    new_folder = old_folder.parent / new_name
    if new_folder.exists():
        raise FileExistsError("Target name already exists.")
    old_folder.rename(new_folder)
    meta = read_meta(new_folder)
    meta["model_name"] = new_name
    (new_folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return new_folder

def delete_model_folder(folder: Path):
    shutil.rmtree(folder)


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

# Compact KPI styles — local to this page
st.markdown("""
<style>
/* Scope to KPI container so it stays local */
.kpi-wrap {
  width: 100%;
  /* optional: shrink overall footprint
  max-width: 880px;
  margin: 0 auto;
  */
}

.kpi-bar {
  display: grid;
  gap: 8px; /* was ~14px */
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); /* was 180px */
  width: 100%;
}

.kpi-tile {
  position: relative;
  border-radius: 12px;        /* was 16px */
  padding: 10px 12px;         /* was 16px 18px */
  color: var(--text, #e5e7eb);
  background:
    radial-gradient(120% 140% at 10% -20%, rgba(34,211,238,0.08), transparent 45%),
    radial-gradient(120% 140% at 110% -10%, rgba(167,139,250,0.08), transparent 45%),
    linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
  border: 1px solid var(--stroke, rgba(255,255,255,0.10));
  box-shadow:
    0 10px 24px rgba(0,0,0,0.35),
    inset 0 1px 0 rgba(255,255,255,0.04);
  overflow: hidden;
}

.kpi-tile::before {
  content:"";
  position:absolute; inset:0;
  border-radius: 12px;
  padding: 0.5px;  /* thinner rim */
  background: linear-gradient(135deg, rgba(34,211,238,0.35), rgba(167,139,250,0.35));
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events:none;
}

.kpi-tile::after {
  content:"";
  position:absolute; inset:0;
  border-radius: 12px;
  background: radial-gradient(120% 120% at 50% 120%, rgba(0,0,0,0.14), transparent 50%);
  pointer-events:none;
}

.kpi-label {
  font-size: 11px;    /* was 12px */
  color: var(--muted, #93a4b8);
  letter-spacing: .02em;
  margin-bottom: 4px; /* was 6px */
}

.kpi-value {
  font-size: 20px;    /* was 26px */
  font-weight: 900;
  letter-spacing: -0.02em;
  color: #f1f5fb;
  text-shadow: 0 0.5px 0 rgba(0,0,0,0.25);
}

/* Even tighter on small screens */
@media (max-width: 720px) {
  .kpi-bar {
    gap: 6px;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
  }
  .kpi-tile { padding: 8px 10px; }
  .kpi-value { font-size: 18px; }
}
</style>
""", unsafe_allow_html=True)


col_left, col_right = st.columns([2, 1])

with col_left:
    # Gesture selection
    all_gestures = list_gestures(DATA_DIR)
    if not all_gestures:
        st.error(f"No gesture folders found under {DATA_DIR}/")
        st.stop()

    st.subheader("Select gestures to include")
    selected = st.multiselect(
        "Gestures", options=all_gestures, default=all_gestures,
        help="Choose which gesture folders to include in training."
    )

    model_name_input = st.text_input(
        "Model name",
        value="SignLSTM",
        help="A human-friendly base name; final folder will include a timestamp for uniqueness."
    )

    notes = st.text_area("Notes (optional)", placeholder="e.g., camera index=0, seq_len=60, FPS=30")

    train_btn = st.button("Train model", type="primary", use_container_width=True)

    if train_btn:
        if len(selected) < 2:
            st.warning("Select at least 2 gestures to train a classifier.")
            st.stop()

        with st.spinner("Loading dataset..."):
            X, y, label_map = load_dataset(selected)
        if X.size == 0:
            st.error("No valid .npy sequences found for selected gestures.")
            st.stop()

        st.info(f"Samples: {X.shape[0]} | Sequence shape: {X.shape[1:]} | Classes: {len(label_map)}")

        with st.spinner("Training LSTM..."):
            model, scaler, metrics = train_model(X, y)

        # Persist artifacts
        out_dir = unique_model_dir(model_name_input)
        meta = {
            "model_name": out_dir.name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "labels": {v: k for k, v in label_map.items()},
            "label_map": label_map,
            "metrics": metrics,
            "data_source": str(DATA_DIR.resolve()),
            "notes": notes,
        }
        save_artifacts(out_dir, model, scaler, meta)

        st.success(f"Saved model to {out_dir}")

with col_right:
    st.subheader("Manage saved models")
    folders = load_models_index()
    if not folders:
        st.caption("No models saved yet.")
    else:
        choices = {f.name: f for f in folders}
        sel = st.selectbox("Select a model folder", options=list(choices.keys()))

        sel_folder = choices[sel]
        meta = read_meta(sel_folder)
        st.write("Accuracy Metrics")
        
        m = meta.get("metrics", {})
        val_acc_pct = fmt_pct(m.get("val_accuracy", None))
        test_acc_pct = fmt_pct(m.get("test_accuracy", None))
        classes_cnt = len(meta.get("labels", {}))

        st.markdown(f"""
        <div class="kpi-wrap">
            <div class="kpi-bar">
                <div class="kpi-tile">
                    <div class="kpi-label">Validation</div>
                    <div class="kpi-value">{val_acc_pct}</div>
                </div>
                <div class="kpi-tile">
                    <div class="kpi-label">Test</div>
                    <div class="kpi-value">{test_acc_pct}</div>
                </div>
                <div class="kpi-tile">
                    <div class="kpi-label">Classes</div>
                    <div class="kpi-value">{classes_cnt}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div></div>', unsafe_allow_html=True)
        labels = meta.get("labels", {})

        with st.expander("Labels", expanded=False):
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


        new_name = st.text_input("Rename (base name)", value=slugify(sel.split("-")[0]))
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Rename", use_container_width=True):
                try:
                    new_folder = rename_model_folder(sel_folder, new_name)
                    # Persist a toast message for next render
                    st.session_state["op_status"] = {
                        "type": "success",
                        "msg": f"Renamed to {new_folder.name}."
                    }
                    # Optionally remember selection; UI list will change after rename
                    st.session_state["last_model_sel"] = new_folder.name
                    st.rerun()
                except Exception as e:
                    st.session_state["op_status"] = {
                        "type": "error",
                        "msg": f"Rename failed: {e}"
                    }
                    st.rerun()

        with col_b:
            if st.button("Delete", use_container_width=True, type="secondary"):
                try:
                    delete_model_folder(sel_folder)
                    st.session_state["op_status"] = {
                        "type": "warning",
                        "msg": f"Deleted {sel_folder.name}."
                    }
                    # After delete, clear last selection so selectbox defaults to first item
                    st.session_state["last_model_sel"] = None
                    st.rerun()
                except Exception as e:
                    st.session_state["op_status"] = {
                        "type": "error",
                        "msg": f"Delete failed: {e}"
                    }
                    st.rerun()
