import io
import json
import os
import tempfile
import re
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple

# OpenCV is optional on Streamlit Community Cloud sometimes (wheel/build issues).
# We fall back to PIL/numpy for image preprocessing, and we disable video mode
# if OpenCV isn't available.
try:
    import cv2  # type: ignore

    CV2_AVAILABLE = True
except Exception:
    cv2 = None  # type: ignore
    CV2_AVAILABLE = False
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# TensorFlow may not be available on the Streamlit runtime (e.g. Python 3.13
# without TF wheels). We treat it as an optional dependency and show a clear
# runtime error if it's missing.
try:
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    tf = None  # type: ignore
    vgg16_preprocess_input = None  # type: ignore
    load_model = None  # type: ignore

try:
    import mediapipe as mp  # type: ignore

    MP_AVAILABLE = True
except Exception:
    mp = None  # type: ignore
    MP_AVAILABLE = False


APP_TITLE = "Emotion & Hand Gesture Recognition"
DEFAULT_MODEL_NAME = "best_hand_gesture_model.keras"
EMOTION_MODEL_NAME = "best_emotion_model.keras"
DEFAULT_IMG_SIZE = 128
EMOTION_LABELS = [
    "angry",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]
FALLBACK_LABELS = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]


def _inject_professional_theme(*, hide_sidebar: bool = False) -> None:
    """
    Streamlit-native UI polish: typography, spacing, primary actions, cards.
    hide_sidebar: use on the landing screen so an empty sidebar is not shown.
    """
    sidebar_rule = (
        "section[data-testid='stSidebar'] { display: none !important; }"
        " div[data-testid='collapsedControl'] { display: none !important; }"
        if hide_sidebar
        else ""
    )
    st.markdown(
        f"""
        <style>
        {sidebar_rule}
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
        html, body, [class*="css"] {{
            font-family: "DM Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
        }}
        .stApp {{
            background: linear-gradient(165deg, #f8fafc 0%, #eef2f7 42%, #f1f5f9 100%);
        }}
        .main .block-container {{
            padding-top: 1.75rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }}
        h1 {{
            font-weight: 700 !important;
            letter-spacing: -0.03em;
            color: #0f172a !important;
            font-size: 2rem !important;
            line-height: 1.2 !important;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 0.65rem;
            margin-bottom: 0.35rem !important;
        }}
        h2, h3 {{
            color: #1e293b !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }}
        [data-testid="stCaptionContainer"] {{
            color: #64748b !important;
            font-size: 1rem !important;
        }}
        div[data-testid="stRadio"] label {{
            font-weight: 500;
            color: #334155;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 55%, #3b82f6 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.65rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
        }}
        div[data-testid="stButton"] > button[kind="secondary"] {{
            border-radius: 10px !important;
            font-weight: 500 !important;
            border-color: #cbd5e1 !important;
            color: #475569 !important;
        }}
        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stNumberInput"] label,
        div[data-testid="stCheckbox"] label {{
            font-weight: 600;
            color: #334155;
            font-size: 0.9rem;
        }}
        [data-baseweb="select"] > div {{
            border-radius: 8px !important;
        }}
        .welcome-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin: 1.25rem 0 1.5rem 0;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }}
        .welcome-pill {{
            display: inline-block;
            background: #eff6ff;
            color: #1d4ed8;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            padding: 0.35rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.75rem;
        }}
        .module-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: #f1f5f9;
            color: #475569;
            font-size: 0.875rem;
            font-weight: 500;
            padding: 0.4rem 0.85rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0 0 1rem 0;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border-right: 1px solid #e2e8f0;
        }}
        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


_LABEL_PREFIX_RE = re.compile(r"^\s*\d+_(.+?)\s*$")


def format_label_for_display(label: str) -> str:
    """
    Convert numeric-prefixed labels like `01_palm` into folder-style labels like `palm`.
    """
    raw = str(label).strip()
    match = _LABEL_PREFIX_RE.match(raw)
    return match.group(1) if match else raw


def _find_model_path(model_name: str) -> Optional[Path]:
    cwd = Path(".")
    exact = cwd / model_name
    if exact.exists():
        return exact

    # Fallback: return the first matching file if present.
    keras_files = sorted(cwd.glob("*.keras"))
    for p in keras_files:
        if p.name == model_name:
            return p

    # Last resort: if there is exactly one .keras file, use it.
    if len(keras_files) == 1:
        return keras_files[0]
    return None


def _scan_labels_from_dataset(dataset_root: Path) -> List[str]:
    if not dataset_root.exists() or not dataset_root.is_dir():
        return []

    labels = set()
    for user_dir in dataset_root.iterdir():
        if not user_dir.is_dir():
            continue
        for gesture_dir in user_dir.iterdir():
            if gesture_dir.is_dir():
                labels.add(gesture_dir.name)

    def sort_key(name: str) -> Tuple[int, str]:
        try:
            prefix = int(name.split("_", 1)[0])
        except Exception:
            prefix = 9999
        return (prefix, name)

    return sorted(labels, key=sort_key)


def _load_labels() -> List[str]:
    labels_json = Path("labels.json")
    if labels_json.exists():
        try:
            with labels_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "class_names" in data:
                data = data["class_names"]
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass

    dataset_root = Path(os.environ.get("DATASET_ROOT", "leapGestRecog"))
    scanned = _scan_labels_from_dataset(dataset_root)
    if scanned:
        return scanned

    return FALLBACK_LABELS


@st.cache_resource
def get_model() -> Any:
    if tf is None or load_model is None:
        raise ImportError(
            "TensorFlow is not available in this Streamlit environment. "
            "Switch Streamlit's Python runtime to <= 3.12 and redeploy, "
            "or install TensorFlow."
        )
    model_path = _find_model_path(DEFAULT_MODEL_NAME)
    if model_path is None:
        raise FileNotFoundError(
            f"Model not found. Put '{DEFAULT_MODEL_NAME}' (or any .keras file) in this folder."
        )
    return load_model(model_path)


@st.cache_resource
def get_emotion_model() -> Any:
    if tf is None or load_model is None:
        raise ImportError(
            "TensorFlow is not available in this Streamlit environment. "
            "Switch Streamlit's Python runtime to <= 3.12 and redeploy, "
            "or install TensorFlow."
        )
    model_path = _find_model_path(EMOTION_MODEL_NAME)
    if model_path is None:
        raise FileNotFoundError(
            f"Model not found. Put '{EMOTION_MODEL_NAME}' (or any .keras file) in this folder."
        )
    return load_model(model_path)


@st.cache_data
def get_labels() -> List[str]:
    return _load_labels()


def preprocess_image(
    img_rgb: np.ndarray,
    img_size: int = DEFAULT_IMG_SIZE,
    use_vgg_preprocess: bool = False,
) -> np.ndarray:
    if CV2_AVAILABLE:
        resized = cv2.resize(img_rgb, (img_size, img_size))  # type: ignore[union-attr]
    else:
        # PIL resize fallback (keeps the app runnable without OpenCV).
        resized = np.array(Image.fromarray(img_rgb).resize((img_size, img_size)))
    arr = resized.astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    if use_vgg_preprocess:
        if vgg16_preprocess_input is None:
            raise ImportError("VGG16 preprocess_input requires TensorFlow.")
        arr = vgg16_preprocess_input(arr)  # type: ignore[misc]
    else:
        arr = arr / 255.0
    return arr


def get_img_size() -> int:
    return int(st.session_state.get("img_size", DEFAULT_IMG_SIZE))


def predict_image(
    img_rgb: np.ndarray,
    model: Any,
    class_names: List[str],
    img_size: Optional[int] = None,
    use_vgg_preprocess: bool = False,
) -> Tuple[str, float, np.ndarray]:
    resolved_size = img_size if img_size is not None else get_img_size()
    batch = preprocess_image(img_rgb, img_size=resolved_size, use_vgg_preprocess=use_vgg_preprocess)
    probs = model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    conf = float(probs[idx])
    return label, conf, probs


def detect_vgg_preprocess_from_model(model: Any) -> bool:
    """
    Heuristic: if the model looks like it contains VGG16 conv blocks,
    use `tensorflow.keras.applications.vgg16.preprocess_input`.
    """
    try:
        for layer in getattr(model, "layers", []):
            name = str(getattr(layer, "name", "")).lower()
            if "block1_conv1" in name or "block2_conv1" in name or "vgg" in name:
                return True
    except Exception:
        pass
    return False


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _pad_bbox(
    x1: int, y1: int, x2: int, y2: int, pad_ratio: float, w: int, h: int
) -> Tuple[int, int, int, int]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    return _clamp_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, w, h)


@st.cache_resource
def get_face_detector() -> Any:
    if not MP_AVAILABLE or mp is None:
        raise ImportError("MediaPipe is not installed. Multi-person Emotion detection requires MediaPipe.")
    # model_selection=1 tends to work well for a broad range of face sizes.
    return mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


@st.cache_resource
def get_hands_detector_static() -> Any:
    if not MP_AVAILABLE or mp is None:
        raise ImportError("MediaPipe is not installed. Multi-person Hand detection requires MediaPipe.")
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=0.5,
    )


def detect_faces(img_rgb: np.ndarray, pad_ratio: float = 0.25, max_faces: int = 6) -> List[Tuple[int, int, int, int]]:
    if not MP_AVAILABLE:
        return []
    detector = get_face_detector()
    h, w = img_rgb.shape[:2]
    results = detector.process(img_rgb)
    bboxes: List[Tuple[int, int, int, int]] = []
    detections = getattr(results, "detections", None) or []
    for det in detections[:max_faces]:
        rb = det.location_data.relative_bounding_box
        x1 = int(rb.xmin * w)
        y1 = int(rb.ymin * h)
        x2 = int((rb.xmin + rb.width) * w)
        y2 = int((rb.ymin + rb.height) * h)
        bboxes.append(_pad_bbox(x1, y1, x2, y2, pad_ratio, w, h))
    return bboxes


def detect_hands(img_rgb: np.ndarray, pad_ratio: float = 0.25, max_hands: int = 4) -> List[Tuple[int, int, int, int]]:
    if not MP_AVAILABLE:
        return []
    detector = get_hands_detector_static()
    h, w = img_rgb.shape[:2]
    results = detector.process(img_rgb)
    bboxes: List[Tuple[int, int, int, int]] = []
    landmarks_list = getattr(results, "multi_hand_landmarks", None) or []
    for hand_landmarks in landmarks_list[:max_hands]:
        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        x1 = int(min(xs) * w)
        y1 = int(min(ys) * h)
        x2 = int(max(xs) * w)
        y2 = int(max(ys) * h)
        bboxes.append(_pad_bbox(x1, y1, x2, y2, pad_ratio, w, h))
    return bboxes


def annotate_image(
    img_rgb: np.ndarray,
    detections: List[dict],
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Draw bounding boxes + labels on an RGB image using PIL.
    Each detection: {"bbox": (x1,y1,x2,y2), "label": str, "confidence": float}
    """
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = str(det.get("label", ""))
        conf = float(det.get("confidence", 0.0))
        text = f"{label} ({conf * 100:.1f}%)"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        # Text background for readability.
        text_w = draw.textlength(text, font=font)
        text_h = 12
        tx1 = x1
        ty1 = max(0, y1 - text_h - 2)
        draw.rectangle([tx1, ty1, tx1 + int(text_w) + 4, ty1 + text_h + 2], fill=color)
        draw.text((tx1 + 2, ty1 + 1), text, fill=(0, 0, 0), font=font)
    return np.array(pil_img)


def predict_emotion_multi(
    img_rgb: np.ndarray,
    model: Any,
    use_vgg_preprocess: bool,
) -> List[dict]:
    bboxes = detect_faces(img_rgb)
    if not bboxes:
        bboxes = [(0, 0, img_rgb.shape[1], img_rgb.shape[0])]

    detections: List[dict] = []
    for (x1, y1, x2, y2) in bboxes:
        crop = img_rgb[y1:y2, x1:x2]
        label, conf, _ = predict_image(crop, model, EMOTION_LABELS, use_vgg_preprocess=use_vgg_preprocess)
        detections.append({"bbox": (x1, y1, x2, y2), "label": label, "confidence": conf})
    return detections


def predict_hand_multi(
    img_rgb: np.ndarray,
    model: Any,
    class_names: List[str],
    use_vgg_preprocess: bool,
) -> List[dict]:
    bboxes = detect_hands(img_rgb)
    if not bboxes:
        bboxes = [(0, 0, img_rgb.shape[1], img_rgb.shape[0])]

    detections: List[dict] = []
    for (x1, y1, x2, y2) in bboxes:
        crop = img_rgb[y1:y2, x1:x2]
        label, conf, _ = predict_image(crop, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        detections.append(
            {
                "bbox": (x1, y1, x2, y2),
                "label": format_label_for_display(label),
                "confidence": conf,
            }
        )
    return detections


def render_prediction_result(label: str, conf: float) -> None:
    display_label = format_label_for_display(label)
    st.success(f"Prediction: {display_label}")
    st.write(f"Confidence: **{conf * 100:.2f}%**")


def handle_single_image(model: Any, class_names: List[str], use_vgg_preprocess: bool) -> None:
    snapshot = st.camera_input("Capture one image (webcam)")
    if snapshot is None:
        return

    image = Image.open(snapshot).convert("RGB")
    img_rgb = np.array(image)

    detections = predict_hand_multi(img_rgb, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
    annotated = annotate_image(img_rgb, detections, color=(0, 200, 0))

    st.image(annotated, caption="Detected hands", use_container_width=True)
    st.write(f"Hands detected: {len(detections)}")
    for i, det in enumerate(detections, start=1):
        st.write(f"{i}. **{det['label']}** - {det['confidence'] * 100:.2f}%")


def handle_multiple_images(model: Any, class_names: List[str], use_vgg_preprocess: bool) -> None:
    items = _collect_multi_rgb_images(
        uploader_key="multi_uploader",
        session_capture_key="hand_multi_captures",
        camera_widget_key="hand_multi_cam",
        upload_prompt="Upload one or more images (hand gestures)",
    )
    if not items:
        st.info("Choose **Capture multiple from camera** or **Upload files**, then add at least one image.")
        return

    st.subheader("Results")
    cols = st.columns(3)
    for i, (name, img_rgb) in enumerate(items):
        detections = predict_hand_multi(img_rgb, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        annotated = annotate_image(img_rgb, detections, color=(0, 200, 0))
        with cols[i % 3]:
            st.image(annotated, caption=name, use_container_width=True)
            if detections:
                best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
                st.write(f"**{best['label']}**")
                st.caption(f"{best['confidence'] * 100:.2f}%")
            else:
                st.caption("No hands detected")


def handle_video(model: Any, class_names: List[str], use_vgg_preprocess: bool) -> None:
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov"], key="video_uploader")
    if uploaded is None:
        return

    if not CV2_AVAILABLE:
        st.error("Video mode requires `opencv-python-headless`. Install it or switch to image mode.")
        return

    sample_every_n = st.slider("Sample every N frames", min_value=5, max_value=60, value=15, step=1)
    max_samples = st.slider("Max sampled frames", min_value=10, max_value=300, value=80, step=10)

    temp_path = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    st.video(temp_path)
    cap = cv2.VideoCapture(temp_path)

    if not cap.isOpened():
        st.error("Could not read video file.")
        return

    frame_id = 0
    sampled = 0
    rows = []
    labels_counter = Counter()

    with st.spinner("Analyzing video..."):
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_id % sample_every_n == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                label, conf, _ = predict_image(
                    frame_rgb,
                    model,
                    class_names,
                    use_vgg_preprocess=use_vgg_preprocess,
                )
                display_label = format_label_for_display(label)
                rows.append({"frame": frame_id, "label": display_label, "confidence": round(conf * 100, 2)})
                labels_counter[display_label] += 1
                sampled += 1
                if sampled >= max_samples:
                    break
            frame_id += 1
        cap.release()

    if not rows:
        st.warning("No sampled frames were analyzed.")
        return

    majority_label, count = labels_counter.most_common(1)[0]
    st.success(f"Overall video prediction: {majority_label} ({count}/{sampled} sampled frames)")
    st.dataframe(rows, use_container_width=True)


def handle_emotion_single_image(model: Any, use_vgg_preprocess: bool) -> None:
    snapshot = st.camera_input("Capture one image (webcam) for emotion detection")
    if snapshot is None:
        return

    image = Image.open(snapshot).convert("RGB")
    img_rgb = np.array(image)
    detections = predict_emotion_multi(img_rgb, model, use_vgg_preprocess=use_vgg_preprocess)
    annotated = annotate_image(img_rgb, detections, color=(200, 0, 0))

    st.image(annotated, caption="Detected faces", use_container_width=True)
    st.write(f"Faces detected: {len(detections)}")
    for i, det in enumerate(detections, start=1):
        st.write(f"{i}. **{det['label']}** - {det['confidence'] * 100:.2f}%")


def handle_emotion_multiple_images(model: Any, use_vgg_preprocess: bool) -> None:
    items = _collect_multi_rgb_images(
        uploader_key="emotion_multi_uploader",
        session_capture_key="emotion_multi_captures",
        camera_widget_key="emotion_multi_cam",
        upload_prompt="Upload one or more images (emotion)",
    )
    if not items:
        st.info("Choose **Capture multiple from camera** or **Upload files**, then add at least one image.")
        return

    st.subheader("Emotion results")
    cols = st.columns(3)
    for i, (name, img_rgb) in enumerate(items):
        detections = predict_emotion_multi(img_rgb, model, use_vgg_preprocess=use_vgg_preprocess)
        annotated = annotate_image(img_rgb, detections, color=(200, 0, 0))
        with cols[i % 3]:
            st.image(annotated, caption=name, use_container_width=True)
            if detections:
                best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
                st.write(f"**{best['label']}**")
                st.caption(f"{best['confidence'] * 100:.2f}%")
            else:
                st.caption("No faces detected")


def _collect_multi_rgb_images(
    *,
    uploader_key: str,
    session_capture_key: str,
    camera_widget_key: str,
    upload_prompt: str,
) -> List[Tuple[str, np.ndarray]]:
    """
    For batch mode: either upload multiple files, or capture several photos from the webcam
    (stored in session_state) and return (display_name, RGB array) pairs.
    """
    mode = st.radio(
        "Add images",
        ["Capture multiple from camera", "Upload files"],
        key=f"{uploader_key}_mode",
        horizontal=False,
        help="Camera: take a picture, then add it to your batch. Upload: select several files at once.",
    )

    if mode == "Upload files":
        files = st.file_uploader(
            upload_prompt,
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key=uploader_key,
        )
        if not files:
            return []
        out: List[Tuple[str, np.ndarray]] = []
        for uf in files:
            out.append((uf.name, np.array(Image.open(uf).convert("RGB"))))
        return out

    if session_capture_key not in st.session_state:
        st.session_state[session_capture_key] = []

    st.caption(
        "Take a photo below, then click **Add current photo to batch**. "
        "Repeat to collect multiple images. Use **Clear batch** to start over."
    )
    snapshot = st.camera_input("Camera", key=camera_widget_key)
    bc1, bc2 = st.columns(2)
    with bc1:
        if st.button("Add current photo to batch", key=f"{uploader_key}_add"):
            if snapshot is None:
                st.warning("Capture a photo first, then add it to the batch.")
            else:
                n = len(st.session_state[session_capture_key]) + 1
                st.session_state[session_capture_key].append(
                    {"bytes": snapshot.getvalue(), "name": f"Capture_{n}"}
                )
                st.success(f"Added photo {n} ({len(st.session_state[session_capture_key])} in batch).")
    with bc2:
        if st.button("Clear batch", key=f"{uploader_key}_clear"):
            st.session_state[session_capture_key] = []
            st.rerun()

    out_cam: List[Tuple[str, np.ndarray]] = []
    for block in st.session_state[session_capture_key]:
        img = Image.open(io.BytesIO(block["bytes"])).convert("RGB")
        out_cam.append((block["name"], np.array(img)))
    return out_cam


def main() -> None:
    started = bool(st.session_state.get("started", False))
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="collapsed" if not started else "expanded",
    )
    _inject_professional_theme(hide_sidebar=not started)

    st.title(APP_TITLE)
    st.caption(
        "Classify **facial expressions** and **hand gestures** from single or **multiple** images "
        "(upload files or capture several photos from the camera), with optional face and hand localization."
    )

    if not started:
        st.markdown(
            """
            <div class="welcome-card">
            <div class="welcome-pill">Machine learning · Computer vision</div>
            <p style="margin:0; color:#334155; font-size:1.05rem; line-height:1.6;">
            Choose a module to load the corresponding trained model. Inference runs in your session;
            use the sidebar after launch to tune input size and preprocessing for your checkpoint.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Get started")
        module = st.radio(
            "Select capability",
            ["Emotion Detection", "Hand Gesture Recognition"],
            index=0,
            horizontal=False,
            help="Emotion uses MediaPipe face regions when available; gestures use hand landmarks.",
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Continue", type="primary", use_container_width=True):
                st.session_state["module"] = module
                st.session_state["started"] = True
                st.rerun()
        with c2:
            st.caption("You can return to this screen anytime from **Settings → Back to home**.")
        return

    module = st.session_state.get("module", "Hand Gesture Recognition")
    emoji = "😊" if module == "Emotion Detection" else "✋"
    st.markdown(
        f'<div class="module-badge">{emoji} Active: {module}</div>',
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Inference and environment")
        st.divider()
        with st.expander("How to run locally", expanded=False):
            st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

        st.session_state["img_size"] = st.number_input(
            "Input image size (px)",
            min_value=64,
            max_value=512,
            value=DEFAULT_IMG_SIZE,
            step=16,
            help="Resize dimension for model input (square). Match training size when possible.",
        )

        st.session_state["force_vgg_preprocess"] = st.checkbox(
            "Force VGG16 preprocess_input",
            value=False,
            help="If disabled, preprocessing matches the loaded model (auto-detected from layer names).",
        )
        st.divider()
        if st.button("Back to home", use_container_width=True):
            st.session_state["started"] = False
            st.rerun()

    try:
        if module == "Emotion Detection":
            model = get_emotion_model()
            class_names = EMOTION_LABELS
        else:
            model = get_model()
            class_names = get_labels()
    except Exception as e:
        st.error(str(e))
        st.stop()

    force_vgg = bool(st.session_state.get("force_vgg_preprocess", False))
    auto_vgg = detect_vgg_preprocess_from_model(model)
    use_vgg_preprocess = force_vgg or auto_vgg

    if module == "Emotion Detection":
        input_mode = st.selectbox(
            "Input source",
            ["Single image capture", "Multiple images"],
            help="Single: one camera shot. Multiple: upload several files or add several webcam captures to a batch.",
        )
        if input_mode == "Single image capture":
            handle_emotion_single_image(model, use_vgg_preprocess=use_vgg_preprocess)
        else:
            handle_emotion_multiple_images(model, use_vgg_preprocess=use_vgg_preprocess)
    else:
        input_mode = st.selectbox(
            "Input source",
            ["Single image capture", "Multiple images"],
            help="Single: one camera shot. Multiple: upload several files or add several webcam captures to a batch.",
        )
        if input_mode == "Single image capture":
            handle_single_image(model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        else:
            handle_multiple_images(model, class_names, use_vgg_preprocess=use_vgg_preprocess)


if __name__ == "__main__":
    main()
