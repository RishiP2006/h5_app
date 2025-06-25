import sys
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import re
from huggingface_hub import hf_hub_download, HfApi
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

st.set_page_config(page_title="Drosophila Gender Detection", layout="centered")
st.title("🪰 Drosophila Gender Detection")
st.write("Select a model and upload an image or use live camera.")

HF_REPO_ID = "RishiPTrial/models_h5"

def check_ultralytics():
    try:
        import ultralytics
        version = ultralytics.__version__ if hasattr(ultralytics, "__version__") else "unknown"
        st.info(f"Ultralytics installed, version: {version}")
        return True
    except Exception as e:
        st.warning(f"Ultralytics import failed: {e}")
        return False

_ULTRA_AVAILABLE = check_ultralytics()

@st.cache_data(show_spinner=False)
def list_hf_models():
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID)
        return [f for f in files if f.lower().endswith((".pt", ".h5", ".pth")) and not f.startswith(".")]
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def build_models_info():
    files = list_hf_models()
    info = {}
    for fname in files:
        input_size = 224
        if "inceptionv3" in fname.lower():
            input_size = 299
        lower = fname.lower()
        if lower.endswith(".pt"):
            info[fname] = {"type": "detection", "framework": "yolo"}
        elif lower.endswith(".h5"):
            info[fname] = {"type": "classification", "framework": "keras", "input_size": input_size}
        elif fname == "model_final.pth":
            info[fname] = {"type": "classification", "framework": "torch_custom", "input_size": input_size}
        elif lower.endswith(".pth"):
            info[fname] = {"type": "classification", "framework": "torch", "input_size": input_size}
    return info

MODELS_INFO = build_models_info()
if not MODELS_INFO:
    st.error(f"No model files found in HF repo {HF_REPO_ID}")

def load_model_final_pth(path):
    import torch
    import torch.nn as nn
    from torchvision import models as _torch_models
    model = _torch_models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_model_from_hf(name, info):
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=name)
    except Exception as e:
        st.error(f"Error downloading {name}: {e}")
        return None

    fw = info.get("framework")
    try:
        if fw == "keras":
            # Attempt standalone keras first
            try:
                import keras
                model = keras.models.load_model(path, compile=False)
                return model
            except Exception:
                pass
            # Fallback to tensorflow.keras with custom_objects if needed
            try:
                import tensorflow as tf
                custom_objects = {}
                lname = name.lower()
                # Add preprocess_input based on model name
                if "resnet50" in lname:
                    from tensorflow.keras.applications.resnet50 import preprocess_input
                    custom_objects["preprocess_input"] = preprocess_input
                elif "inceptionv3" in lname or "inception_v3" in lname:
                    from tensorflow.keras.applications.inception_v3 import preprocess_input
                    custom_objects["preprocess_input"] = preprocess_input
                # You can add more architectures here as needed
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                return model
            except Exception as e:
                st.error(f"Failed loading Keras model {name}: {e}")
                return None

        if fw == "torch_custom":
            try:
                import torch
                return load_model_final_pth(path)
            except Exception as e:
                st.error(f"Failed loading custom PyTorch model {name}: {e}")
                return None

        if fw == "torch":
            try:
                import torch
                m = torch.load(path, map_location="cpu")
                m.eval()
                return m
            except Exception as e:
                st.error(f"Failed loading PyTorch model {name}: {e}")
                return None

        if fw == "yolo":
            if not _ULTRA_AVAILABLE:
                st.error("Ultralytics YOLO not installed; cannot load detection model.")
                return None
            try:
                from ultralytics import YOLO
                return YOLO(path)
            except Exception as e:
                st.error(f"Failed loading YOLO model {name}: {e}")
                return None

    except Exception as e:
        st.error(f"Failed loading {name}: {e}")
        return None

    st.error(f"Unsupported framework for {name}")
    return None

# ---------------- Inference helpers ----------------

def preprocess_image_pil(pil_img: Image.Image, size: int):
    arr = pil_img.resize((size, size))
    arr = np.asarray(arr).astype(np.float32) / 255.0
    return arr

def classify(model, img_array: np.ndarray):
    x = np.expand_dims(img_array, axis=0)
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return model.predict(x)
    except Exception:
        pass
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                x_t = torch.tensor(x).permute(0,3,1,2).float()
                out = model(x_t)
                return out.cpu().numpy()
    except Exception:
        pass
    st.error("Unknown model type for prediction.")
    return None

def interpret_classification(preds):
    if preds is None:
        return None, None
    arr = np.asarray(preds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        exps = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        label = ["Male", "Female"][idx]
        return label, float(probs[0][idx])
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1/(1+np.exp(-val)) if (val < 0 or val > 1) else val
        label = "Female" if prob >= 0.5 else "Male"
        conf = prob if label == "Female" else 1-prob
        return label, conf
    st.warning(f"Unexpected prediction shape: {arr.shape}")
    return None, None

def detect_yolo(model, pil_img: Image.Image):
    try:
        arr = np.array(pil_img.convert("RGB"))
        results = model.predict(source=arr)
    except Exception as e:
        st.error(f"YOLO inference failed: {e}")
        return []
    detections = []
    for res in results:
        for b in res.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            try:
                coords = tuple(map(int, b.xyxy[0].cpu().numpy()))
            except Exception:
                coords = tuple(map(int, b.xyxy[0]))
            name = model.names.get(cls, str(cls)) if hasattr(model, 'names') else str(cls)
            detections.append((name, conf, coords))
    return detections

class GenderDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.info = MODELS_INFO[model_name]

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="rgb24")
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if self.model is not None:
            if self.info.get("type") == "classification":
                size = self.info.get("input_size", 224)
                arr = preprocess_image_pil(pil, size)
                preds = classify(self.model, arr)
                label, prob = interpret_classification(preds)
                if label:
                    draw.text((10, 10), f"{label} ({prob:.1%})", fill="red")
            else:
                dets = detect_yolo(self.model, pil)
                for name, conf, (x1, y1, x2, y2) in dets:
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                    draw.text((x1, max(y1-10, 0)), f"{name} {conf:.2f}", fill="green")
        return av.VideoFrame.from_ndarray(np.array(pil), format="rgb24")

def safe_label(name):
    return re.sub(r"[^\w\s.-]", "_", name)

safe_to_real = {safe_label(n): n for n in MODELS_INFO}
choice = st.selectbox("Select model", list(safe_to_real.keys())) if MODELS_INFO else None
model_name = safe_to_real.get(choice)
model = None
if model_name:
    info = MODELS_INFO[model_name]
    model = load_model_from_hf(model_name, info)

st.markdown("---")
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file and model is not None:
    pil_img = Image.open(img_file).convert("RGB")
    st.image(pil_img, use_column_width=True)
    info = MODELS_INFO[model_name]
    if info.get("type") == "classification":
        arr = preprocess_image_pil(pil_img, info.get("input_size", 224))
        preds = classify(model, arr)
        label, prob = interpret_classification(preds)
        if label:
            st.success(f"Prediction: {label} ({prob:.1%})")
    else:
        disp = pil_img.copy()
        draw = ImageDraw.Draw(disp)
        dets = detect_yolo(model, pil_img)
        male_count = female_count = 0
        for name, conf, box in dets:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, max(y1-10, 0)), f"{name} {conf:.2f}", fill="green")
            if name.lower() == "male":
                male_count += 1
            elif name.lower() == "female":
                female_count += 1
        st.image(disp, use_column_width=True)
        if dets:
            st.info(f"Detected Males: {male_count}, Females: {female_count}")

st.markdown("---")
st.subheader("📸 Live Camera Gender Detection")
if model is not None:
    ctx = webrtc_streamer(
        key="live-gender-detect",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=GenderDetectionProcessor,
        async_processing=True,
    )
else:
    st.warning("Please select a model first.")

st.markdown("---")
st.write("**Notes:**")
st.write(f"- Models from HF: {HF_REPO_ID}")
st.write("- Classification assumes binary output.")
st.write("- YOLO models output bounding boxes with class names.")
st.write("- Live camera uses PIL for drawing, no cv2 needed.")
