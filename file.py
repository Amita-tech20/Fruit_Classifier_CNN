import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Fruits Lens", page_icon="🌼", layout="centered")
st.title("🌼 Fruits Lens — CNN Classifier (Upload or Camera)")

FRUITS_NAMES = ['Apple', 'Banana', 'Grape', 'Mango']
IMG_SIZE = (180, 180)
CONF_THRESHOLD = 0.70  # confidence threshold for warning

# ---------- Load model ----------
MODEL_PATH = r'C:\Users\Ankita\Desktop\Fruit_Classifier_CNN\Fruit_Recog_Model.h5'
WEIGHTS_PATH = r'C:\Users\Ankita\Desktop\Fruit_Classifier_CNN\Fruit_Recog_Model.weights.h5'

@st.cache_resource(show_spinner=False)
def load_fruit_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    model = keras.models.load_model(MODEL_PATH)
    if os.path.exists(WEIGHTS_PATH):
        model.load_weights(WEIGHTS_PATH)
    return model

model = load_fruit_model()

# ---------- Prediction helpers ----------
def predict_topk(file_like, k=3):
    img = tf.keras.utils.load_img(file_like, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    # x = x / 255.0  # Uncomment if your model expects normalized input
    x = tf.expand_dims(x, 0)
    preds = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(preds).numpy()
    idxs = np.argsort(probs)[::-1][:k]
    return [(FRUITS_NAMES[i], float(probs[i])) for i in idxs]

def render_predictions(title, preds):
    st.subheader(title)
    best_label, best_p = preds[0]

    if best_p < CONF_THRESHOLD:
        st.error(f"❌ This doesn't seem to be a known flower from the trained set.\n"
                 f"Highest match: {best_label} ({best_p*100:.2f}%)")
    else:
        st.success(f"Prediction: **{best_label}** ({best_p*100:.2f}%)")

    st.write("Top results:")
    for label, p in preds:
        st.write(f"- {label}: {p*100:.2f}%")

# ---------- Tabs for Upload vs Camera ----------
tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Use camera"])

with tab_upload:
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", width=260)
        preds = predict_topk(uploaded, k=3)
        render_predictions("Upload Prediction", preds)

with tab_camera:
    cam_img = st.camera_input("Take a photo")
    if cam_img is not None:
        st.image(cam_img, caption="Captured image", width=260)
        preds = predict_topk(cam_img, k=3)
        render_predictions("Camera Prediction", preds)

st.caption("Tip: Confidence below 70% will trigger an 'unknown object' warning.")
