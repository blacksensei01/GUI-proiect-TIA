import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶", layout="centered")
st.title("🐶 Stanford Dogs – Dog Breed Classifier")
st.write("Încarcă o imagine cu un câine și îți arăt top rasele prezise de model.")

MODEL_PATH = "dog_breed_mobilenetv2.keras"
LABELS_PATH = "class_names.json"
IMG_SIZE = (224, 224)

preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

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

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)  # float32, RGB
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess(arr)
    return arr

def predict_topk(model, class_names, img: Image.Image, k=5):
    x = preprocess_pil(img)
    probs = model.predict(x, verbose=0)[0]
    idx = np.argsort(probs)[-k:][::-1]
    return [(class_names[i], float(probs[i])) for i in idx]

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

        st.table({
            "Rasă": [pretty_label(r) for r, _ in top5],
            "Probabilitate": [f"{p*100:.2f}%" for _, p in top5],
        })

        chart_data = {"Rasă": [pretty_label(r) for r, _ in top5],
                      "Probabilitate": [p for _, p in top5]}
        st.bar_chart(chart_data, x="Rasă", y="Probabilitate")

    st.divider()
    st.caption(f"Clase: {len(class_names)} | Model: {MODEL_PATH}")

    # Bonus: "confidence" simplu
    best_label, best_p = top5[0]
    if best_p < 0.25:
        st.warning("⚠️ Modelul nu este foarte sigur (probabilitate mică). Încearcă o imagine mai clară, cu câinele mai mare în cadru.")
else:
    st.info("Încarcă o imagine ca să vezi predicțiile.")
