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
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # type: ignore
    webrtc_streamer = None  # type: ignore


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


@st.cache_resource
def get_hands_detector_stream() -> Any:
    if not MP_AVAILABLE or mp is None:
        raise ImportError("MediaPipe is not installed. Multi-person Hand detection requires MediaPipe.")
    return mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
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


def detect_hands_stream(img_rgb: np.ndarray, pad_ratio: float = 0.25, max_hands: int = 4) -> List[Tuple[int, int, int, int]]:
    if not MP_AVAILABLE:
        return []
    detector = get_hands_detector_stream()
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
    files = st.file_uploader(
        "Upload multiple images",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="multi_uploader",
    )
    if not files:
        return

    st.subheader("Results")
    cols = st.columns(3)
    for i, uploaded in enumerate(files):
        image = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(image)
        detections = predict_hand_multi(img_rgb, model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        annotated = annotate_image(img_rgb, detections, color=(0, 200, 0))
        with cols[i % 3]:
            st.image(annotated, caption=uploaded.name, use_container_width=True)
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


def predict_hand_multi_stream(
    img_rgb: np.ndarray,
    model: Any,
    class_names: List[str],
    use_vgg_preprocess: bool,
) -> List[dict]:
    bboxes = detect_hands_stream(img_rgb)
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


class HandVideoProcessor(VideoProcessorBase):
    def __init__(self, sample_every_n: int = 15) -> None:
        self.sample_every_n = max(1, int(sample_every_n))
        self.frame_idx = 0
        self.last_annotated_bgr: Optional[np.ndarray] = None
        self.last_detections: List[dict] = []
        self.last_label = ""
        self.last_conf = 0.0
        self.error_msg: Optional[str] = None

        try:
            self.model = get_model()
            self.class_names = get_labels()
            force_vgg = bool(st.session_state.get("force_vgg_preprocess", False))
            auto_vgg = detect_vgg_preprocess_from_model(self.model)
            self.use_vgg_preprocess = force_vgg or auto_vgg
        except Exception as e:
            self.error_msg = str(e)
            self.model = None
            self.class_names = []
            self.use_vgg_preprocess = False

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        if self.error_msg is not None:
            # Can't run inference: return original frame.
            return frame.from_ndarray(img_bgr, format="bgr24")

        if self.last_annotated_bgr is not None and (self.frame_idx % self.sample_every_n) != 0:
            return frame.from_ndarray(self.last_annotated_bgr, format="bgr24")

        img_rgb = img_bgr[..., ::-1]
        detections = predict_hand_multi_stream(
            img_rgb,
            self.model,
            self.class_names,
            use_vgg_preprocess=self.use_vgg_preprocess,
        )
        annotated_rgb = annotate_image(img_rgb, detections, color=(0, 200, 0))
        annotated_bgr = annotated_rgb[..., ::-1]

        self.last_detections = detections
        if detections:
            best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
            self.last_label = best.get("label", "")
            self.last_conf = float(best.get("confidence", 0.0))

        self.last_annotated_bgr = annotated_bgr
        return frame.from_ndarray(annotated_bgr, format="bgr24")


class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self, sample_every_n: int = 15) -> None:
        self.sample_every_n = max(1, int(sample_every_n))
        self.frame_idx = 0
        self.last_annotated_bgr: Optional[np.ndarray] = None
        self.last_detections: List[dict] = []
        self.last_label = ""
        self.last_conf = 0.0
        self.error_msg: Optional[str] = None

        try:
            self.model = get_emotion_model()
            force_vgg = bool(st.session_state.get("force_vgg_preprocess", False))
            auto_vgg = detect_vgg_preprocess_from_model(self.model)
            self.use_vgg_preprocess = force_vgg or auto_vgg
        except Exception as e:
            self.error_msg = str(e)
            self.model = None
            self.use_vgg_preprocess = False

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        if self.error_msg is not None or self.model is None:
            return frame.from_ndarray(img_bgr, format="bgr24")

        if self.last_annotated_bgr is not None and (self.frame_idx % self.sample_every_n) != 0:
            return frame.from_ndarray(self.last_annotated_bgr, format="bgr24")

        img_rgb = img_bgr[..., ::-1]
        detections = predict_emotion_multi(img_rgb, self.model, use_vgg_preprocess=self.use_vgg_preprocess)
        annotated_rgb = annotate_image(img_rgb, detections, color=(200, 0, 0))
        annotated_bgr = annotated_rgb[..., ::-1]

        self.last_detections = detections
        if detections:
            best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
            self.last_label = best.get("label", "")
            self.last_conf = float(best.get("confidence", 0.0))

        self.last_annotated_bgr = annotated_bgr
        return frame.from_ndarray(annotated_bgr, format="bgr24")


def handle_live_webcam() -> None:
    if not WEBRTC_AVAILABLE:
        st.warning(
            "Live webcam needs `streamlit-webrtc`. Install dependencies from requirements.txt and restart."
        )
        st.info("You can still use snapshot mode below.")

    st.subheader("Live Webcam")
    if WEBRTC_AVAILABLE:
        sample_every_n = st.slider("Run inference every N frames", min_value=1, max_value=30, value=5, step=1)
        ctx = webrtc_streamer(
            key="hand-live",
            video_processor_factory=lambda: HandVideoProcessor(sample_every_n=sample_every_n),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if ctx and ctx.video_processor:
            st.write(
                f"Live: **{ctx.video_processor.last_label}** ({ctx.video_processor.last_conf * 100:.2f}%)"
            )

    st.subheader("Snapshot")
    snapshot = st.camera_input("Take a single photo")
    if snapshot is not None:
        image = Image.open(snapshot).convert("RGB")
        img_rgb = np.array(image)
        st.image(img_rgb, caption="Captured snapshot", use_container_width=True)
        model = get_model()
        labels = get_labels()
        force_vgg = bool(st.session_state.get("force_vgg_preprocess", False))
        auto_vgg = detect_vgg_preprocess_from_model(model)
        use_vgg = force_vgg or auto_vgg
        detections = predict_hand_multi(img_rgb, model, labels, use_vgg_preprocess=use_vgg)
        annotated = annotate_image(img_rgb, detections, color=(0, 200, 0))
        st.image(annotated, caption="Hand detection result", use_container_width=True)
        st.write(f"Hands detected: {len(detections)}")
        for i, det in enumerate(detections, start=1):
            st.write(f"{i}. **{det['label']}** - {det['confidence'] * 100:.2f}%")


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
    files = st.file_uploader(
        "Upload multiple images for emotion detection",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=True,
        key="emotion_multi_uploader",
    )
    if not files:
        return

    st.subheader("Emotion results")
    cols = st.columns(3)
    for i, uploaded in enumerate(files):
        image = Image.open(uploaded).convert("RGB")
        img_rgb = np.array(image)
        detections = predict_emotion_multi(img_rgb, model, use_vgg_preprocess=use_vgg_preprocess)
        annotated = annotate_image(img_rgb, detections, color=(200, 0, 0))
        with cols[i % 3]:
            st.image(annotated, caption=uploaded.name, use_container_width=True)
            if detections:
                best = max(detections, key=lambda d: float(d.get("confidence", 0.0)))
                st.write(f"**{best['label']}**")
                st.caption(f"{best['confidence'] * 100:.2f}%")
            else:
                st.caption("No faces detected")


def handle_emotion_live_webcam() -> None:
    if not WEBRTC_AVAILABLE:
        st.warning(
            "Live webcam needs `streamlit-webrtc`. Install dependencies from requirements.txt and restart."
        )
        return

    st.subheader("Live Webcam (Emotion)")
    sample_every_n = st.slider("Run inference every N frames", min_value=1, max_value=30, value=5, step=1)
    ctx = webrtc_streamer(
        key="emotion-live",
        video_processor_factory=lambda: EmotionVideoProcessor(sample_every_n=sample_every_n),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx and ctx.video_processor:
        faces = getattr(ctx.video_processor, "last_detections", None) or []
        face_count = len(faces)
        st.write(
            f"Live: **{ctx.video_processor.last_label}** ({ctx.video_processor.last_conf * 100:.2f}%)"
            f" | Faces: {face_count}"
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Predict emotions and hand gestures from image, multiple images, and live webcam.")

    if not st.session_state.get("started", False):
        st.subheader("Welcome")
        module = st.radio(
            "Choose a module",
            ["Emotion Detection", "Hand Gesture Recognition"],
            index=0,
        )
        if st.button("Start", type="primary"):
            st.session_state["module"] = module
            st.session_state["started"] = True
            st.rerun()
        return

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        st.write("Run with:")
        st.code("pip install -r requirements.txt\nstreamlit run app.py")

        st.session_state["img_size"] = st.number_input(
            "Image size",
            min_value=64,
            max_value=512,
            value=DEFAULT_IMG_SIZE,
            step=16,
        )

        st.session_state["force_vgg_preprocess"] = st.checkbox(
            "Use VGG16 preprocess_input (force)",
            value=False,
            help="If off, the app auto-detects preprocessing from the loaded model.",
        )

        if st.button("Back to start"):
            st.session_state["started"] = False
            st.rerun()

    module = st.session_state.get("module", "Hand Gesture Recognition")

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
        input_mode = st.selectbox("Choose input", ["Single image capture", "Multiple image upload", "Live webcam"])
        if input_mode == "Single image capture":
            handle_emotion_single_image(model, use_vgg_preprocess=use_vgg_preprocess)
        elif input_mode == "Multiple image upload":
            handle_emotion_multiple_images(model, use_vgg_preprocess=use_vgg_preprocess)
        else:
            handle_emotion_live_webcam()
    else:
        input_mode = st.selectbox("Choose input", ["Single image capture", "Multiple image upload", "Live webcam"])
        if input_mode == "Single image capture":
            handle_single_image(model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        elif input_mode == "Multiple image upload":
            handle_multiple_images(model, class_names, use_vgg_preprocess=use_vgg_preprocess)
        else:
            handle_live_webcam()


if __name__ == "__main__":
    main()
