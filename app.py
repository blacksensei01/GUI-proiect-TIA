import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")
st.title("🐶 Stanford Dogs – Dog Breed Classifier")

MODEL_PATH = "dog_breed_mobilenetv2.keras"
LABELS_PATH = "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

def pretty_label(lbl: str) -> str:
    if "-" in lbl:
        lbl = lbl.split("-", 1)[1]
    return lbl.replace("_", " ")

# IMPORTANT: NU mai facem preprocess_input aici!
# Modelul tău îl are deja în interior (TrueDivide/Subtract).
#def preprocess_pil(img: Image.Image) -> np.ndarray:
   # img = img.convert("RGB").resize(IMG_SIZE)
    #arr = tf.keras.utils.img_to_array(img)  # 0..255
   # arr = np.expand_dims(arr, axis=0)
    #return arr

def predict_topk(model, class_names, img: Image.Image, k=5):
   # x = preprocess_pil(img)
    probs = model.predict(img, verbose=0)[0]
    idx = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in idx]

try:
    model, class_names = load_model_and_labels()
except Exception as e:
    st.error(f"Nu pot încărca modelul/etichetele: {e}")
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

        st.table({
            "Rasă": [pretty_label(r) for r, _ in top5],
            "Probabilitate": [f"{p*100:.2f}%" for _, p in top5],
        })

        chart_data = {"Rasă": [pretty_label(r) for r, _ in top5],
                      "Probabilitate": [p for _, p in top5]}
        st.bar_chart(chart_data, x="Rasă", y="Probabilitate")

    best_label, best_p = top5[0]
    st.caption(f"Top-1: {pretty_label(best_label)} ({best_p*100:.2f}%)")
else:
    st.info("Încarcă o imagine ca să vezi predicțiile.")

