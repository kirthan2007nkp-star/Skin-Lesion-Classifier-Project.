import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="SkinGuard AI | Skin Lesion Classifier",
    page_icon="🧬",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.75;
    }

    .card {
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    .prediction {
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        border: 2px solid rgba(128,128,128,0.35);
    }

    .prediction h2 {
        margin-bottom: 0.2rem;
    }

    .big-result {
        font-size: 2.2rem;
        font-weight: 700;
    }

    .confidence {
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }

    .footer {
        text-align: center;
        opacity: 0.65;
        padding: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🧬 SkinGuard AI</h1>
    <p>CNN-Based Skin Lesion Classification System</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PROJECT INTRODUCTION
# ---------------------------------------------------------
st.markdown("""
<div class="card">
<h3>🔬 About the Project</h3>
<p>
SkinGuard AI uses a Convolutional Neural Network (CNN) to classify
dermoscopic skin-lesion images into seven categories from the HAM10000
dataset. Upload an image below to test the trained model.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
MODEL_URL = (
    "https://github.com/kirthan2007nkp-star/"
    "Skin-Lesion-Classifier-Project./releases/download/"
    "v1.0/skin_lesion_model.keras"
)

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

class_labels = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

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

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
try:
    model = load_model()
    st.success("✅ CNN model loaded successfully")
except Exception:
    model = None
    st.error("❌ Could not load the model.")

# ---------------------------------------------------------
# UPLOAD SECTION
# ---------------------------------------------------------
st.subheader("📤 Upload Skin-Lesion Image")

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload a dermoscopic skin-lesion image for classification."
)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if uploaded_file and model is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### 🤖 CNN Analysis")

        with st.spinner("Analyzing image..."):
            processed_image = image.resize((224, 224))

            image_array = np.array(
                processed_image,
                dtype=np.float32
            ) / 255.0

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            prediction = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(prediction))
            predicted_class = class_names[predicted_index]
            confidence = float(prediction[predicted_index]) * 100

        st.markdown(
            f"""
            <div class="prediction">
                <h2>Prediction</h2>
                <div class="big-result">
                    {predicted_class.upper()}
                </div>
                <div>
                    {class_labels[predicted_class]}
                </div>
                <div class="confidence">
                    Confidence: <b>{confidence:.2f}%</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Classification Probabilities")

    probability_data = {
        class_names[i].upper(): float(prediction[i]) * 100
        for i in range(len(class_names))
    }

    st.bar_chart(probability_data)

    # -----------------------------------------------------
    # TOP PREDICTIONS
    # -----------------------------------------------------
    st.subheader("🏆 Top Predictions")

    sorted_indices = np.argsort(prediction)[::-1]

    for rank, index in enumerate(sorted_indices[:3], start=1):
        name = class_names[index].upper()
        score = float(prediction[index]) * 100

        st.write(
            f"**{rank}. {name}** — {score:.2f}%"
        )

        st.progress(min(score / 100, 1.0))

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.markdown("---")

st.warning(
    "⚠️ For educational and research purposes only. "
    "This classifier is not a medical diagnosis and should not "
    "be used to make medical decisions."
)

st.markdown("""
<div class="footer">
    <b>SkinGuard AI</b> • CNN-Based Skin Lesion Classifier
</div>
""", unsafe_allow_html=True)    

