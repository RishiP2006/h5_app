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

@st.cache_data
def list_models():
    api = HfApi()
    return [
        f for f in api.list_repo_files(HF_REPO_ID)
        if f.lower().endswith((".h5", ".pt", ".pth"))
    ]

@st.cache_data
def build_model_info():
    info = {}
    for f in list_models():
        name = f.lower()
        if name.endswith(".pt"):
            info[f] = {"type": "detection", "framework": "yolo"}
        elif f == "model_final.pth":
            info[f] = {"type": "classification", "framework": "torch_custom", "input_size": 224}
        elif name.endswith(".pth"):
            info[f] = {"type": "classification", "framework": "torch", "input_size": 224}
        elif name.endswith(".h5"):
            size = 299 if "inceptionv3" in name else 224
            info[f] = {"type": "classification", "framework": "keras", "input_size": size}
    return info

MODELS_INFO = build_model_info()

@st.cache_resource
def load_model(name, info):
    path = hf_hub_download(HF_REPO_ID, filename=name)
    fw = info["framework"]

    if fw == "keras":
        try:
            import keras
            from keras.utils import custom_object_scope

            custom_objects = {}
            lname = name.lower()
            if "resnet" in lname:
                from keras.applications.resnet50 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input
            elif "inceptionv3" in lname:
                from keras.applications.inception_v3 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input
            elif "mobilenetv2" in lname:
                from keras.applications.mobilenet_v2 import preprocess_input
                custom_objects["preprocess_input"] = preprocess_input

            with custom_object_scope(custom_objects):
                model = keras.models.load_model(path, compile=False)
            return model
        except Exception as e:
            st.error(f"Failed loading Keras model {name}: {e}")
            return None

    if fw == "torch_custom":
        import torch
        import torch.nn as nn
        from torchvision import models
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict.get("model", state_dict), strict=False)
        model.eval()
        return model

    if fw == "torch":
        import torch
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model

    if fw == "yolo":
        from ultralytics import YOLO
        return YOLO(path)

    return None

def preprocess_image(img, size):
    return np.asarray(img.resize((size, size))).astype(np.float32) / 255.0

def classify(model, img):
    arr = np.expand_dims(img, axis=0)
    try:
        import keras
        if isinstance(model, keras.Model):
            return model.predict(arr)
    except Exception:
        pass
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            x = torch.tensor(arr).permute(0,3,1,2).float()
            with torch.no_grad():
                return model(x).cpu().numpy()
    except Exception:
        pass
    return None

def interpret_class(preds):
    if preds is None: return None, None
    arr = np.array(preds)
    if arr.ndim == 2 and arr.shape[1] == 2:
        exps = np.exp(arr - np.max(arr))
        probs = exps / np.sum(exps)
        idx = np.argmax(probs)
        return ["Male", "Female"][idx], float(probs[idx])
    if arr.ndim == 2 and arr.shape[1] == 1:
        val = float(arr[0][0])
        prob = 1/(1 + np.exp(-val))
        label = "Female" if prob >= 0.5 else "Male"
        return label, prob if label == "Female" else 1 - prob
    return None, None

def detect(model, img):
    arr = np.array(img.convert("RGB"))
    results = model.predict(source=arr)
    dets = []
    for r in results:
        for b in r.boxes:
            name = model.names[int(b.cls[0])]
            coords = list(map(int, b.xyxy[0].cpu().numpy()))
            dets.append((name, float(b.conf[0]), coords))
    return dets

class GenderProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.info = MODELS_INFO[model_name]
    def recv(self, frame):
        img = Image.fromarray(frame.to_ndarray(format="rgb24"))
        draw = ImageDraw.Draw(img)
        if self.info["type"] == "classification":
            arr = preprocess_image(img, self.info["input_size"])
            preds = classify(self.model, arr)
            label, conf = interpret_class(preds)
            if label: draw.text((10, 10), f"{label} ({conf:.0%})", fill="red")
        else:
            for name, conf, box in detect(self.model, img):
                draw.rectangle(box, outline="green", width=2)
                draw.text((box[0], box[1]-10), f"{name} {conf:.2f}", fill="green")
        return av.VideoFrame.from_ndarray(np.array(img), format="rgb24")

# UI
safe_names = {re.sub(r"[^\w.-]", "_", n): n for n in MODELS_INFO}
model_display = st.selectbox("Select Model", list(safe_names))
model_name = safe_names[model_display]
model = load_model(model_name, MODELS_INFO[model_name])

# Image upload
st.subheader("📷 Upload Image")
img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if img_file and model:
    pil = Image.open(img_file).convert("RGB")
    st.image(pil, use_column_width=True)
    info = MODELS_INFO[model_name]
    if info["type"] == "classification":
        arr = preprocess_image(pil, info["input_size"])
        pred = classify(model, arr)
        label, conf = interpret_class(pred)
        if label:
            st.success(f"Prediction: {label} ({conf:.1%})")
    else:
        img = pil.copy()
        draw = ImageDraw.Draw(img)
        counts = {"male": 0, "female": 0}
        for name, conf, box in detect(model, pil):
            draw.rectangle(box, outline="green", width=2)
            draw.text((box[0], box[1]-10), f"{name} {conf:.2f}", fill="green")
            counts[name.lower()] += 1
        st.image(img)
        st.info(f"Detected: {counts['male']} males, {counts['female']} females")

# Live camera
st.subheader("📸 Live Camera Detection")
if model:
    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=GenderProcessor,
        async_processing=True,
    )

st.markdown("---")
st.write("- Models from:", HF_REPO_ID)
