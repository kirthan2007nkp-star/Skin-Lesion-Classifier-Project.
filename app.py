import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

st.title("Skin Lesion Classifier")
st.write("Upload a skin-lesion image for classification.")

MODEL_URL = "https://github.com/kirthan2007nkp-star/Skin-Lesion-Classifier-Project./releases/download/v1.0/skin_lesion_model.keras"
MODEL_PATH = "skin_lesion_model.keras"

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as response:
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    model = None
    st.error("Could not load the model.")

uploaded_file = st.file_uploader(
    "Upload a skin-lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((224, 224))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    st.success(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

st.caption("For educational purposes only. This tool is not a medical diagnosis.")
