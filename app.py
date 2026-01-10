import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------
# CONFIG UI
# ---------------------------
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")
st.title("🐶 Stanford Dogs – Dog Breed Classifier")
st.write("Încarcă o imagine cu un câine și îți arăt top rasele prezise de model.")

MODEL_PATH = "dog_breed_mobilenetv2.keras"
LABELS_PATH = "class_names.json"
IMG_SIZE = (224, 224)

# ---------------------------
# LOAD MODEL + LABELS (cached)
# ---------------------------
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def preprocess_pil(img: Image.Image) -> np.ndarray:
    # RGB, resize
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)

    # MobileNetV2 preprocess_input: scale to [-1, 1]
    arr = (arr / 127.5) - 1.0

    # add batch dim
    return np.expand_dims(arr, axis=0)

def predict_topk(model, class_names, img: Image.Image, k=5):
    x = preprocess_pil(img)
    probs = model.predict(x, verbose=0)[0]  # shape: (num_classes,)
    idx = np.argsort(probs)[-k:][::-1]
    results = [(class_names[i], float(probs[i])) for i in idx]
    return results

# ---------------------------
# MAIN
# ---------------------------
try:
    model, class_names = load_model_and_labels()
except Exception as e:
    st.error(
        "Nu pot încărca modelul / etichetele.\n\n"
        f"Verifică că există în același folder:\n"
        f"- {MODEL_PATH}\n- {LABELS_PATH}\n\n"
        f"Eroare: {e}"
    )
    st.stop()

uploaded = st.file_uploader("Alege o imagine (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns([1, 1])

if uploaded is not None:
    img = Image.open(uploaded)

    with col1:
        st.subheader("Imaginea încărcată")
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Predicții (Top-5)")
        top5 = predict_topk(model, class_names, img, k=5)

        # tabel
        st.table({
            "Rasă": [r for r, p in top5],
            "Probabilitate": [f"{p*100:.2f}%" for r, p in top5],
        })

        # chart
        chart_data = {"Rasă": [r for r, _ in top5], "Probabilitate": [p for _, p in top5]}
        st.bar_chart(chart_data, x="Rasă", y="Probabilitate")

    st.divider()
    st.caption(f"Model: {MODEL_PATH} | Clase: {len(class_names)}")

else:
    st.info("Încarcă o imagine ca să vezi predicțiile.")
