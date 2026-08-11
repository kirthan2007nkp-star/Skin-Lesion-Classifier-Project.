import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Skin Lesion Classifier")
st.write("Upload a skin-lesion image for classification.")

MODEL_PATH = "skin_lesion_model.keras"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    model = None
    st.warning("Model file is not included in this GitHub repository yet.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    st.write("Prediction:", prediction)
