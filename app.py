import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="SkinGuard AI",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.hero {
    padding: 2.5rem 1rem;
    text-align: center;
    border-radius: 22px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 3.2rem;
    margin-bottom: 5px;
}

.hero p {
    font-size: 1.1rem;
    opacity: .75;
}

.card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128,128,128,.25);
    margin-bottom: 20px;
}

.result {
    text-align: center;
    padding: 25px;
    border-radius: 20px;
    border: 2px solid rgba(128,128,128,.3);
}

.result h1 {
    font-size: 2.5rem;
}

.footer {
    text-align: center;
    opacity: .65;
    padding: 30px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="hero">
    <h1>🧬 SkinGuard AI</h1>
    <p>CNN-Based Skin Lesion Classification System</p>
</div>
""", unsafe_allow_html=True)

# MODEL
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

        with requests.get(
            MODEL_URL,
            stream=True
        ) as response:

            response.raise_for_status()

            with open(MODEL_PATH, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=8192
                ):

                    if chunk:
                        file.write(chunk)

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
    st.success("🟢 CNN Model Ready")

except Exception:
    model = None
    st.error("Model could not be loaded.")

# INTRO
st.markdown("""
<div class="card">

### 🔬 AI Skin Lesion Analysis

Upload a dermoscopic skin-lesion image and the trained
CNN model will classify it into one of seven categories.

The application displays the model's prediction,
confidence and probability distribution.

</div>
""", unsafe_allow_html=True)

# UPLOAD
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose a JPG, JPEG or PNG image",
    type=["jpg", "jpeg", "png"]
)

# PREDICTION
if uploaded_file and model is not None:

    original = Image.open(uploaded_file).convert("RGB")

    st.markdown("---")
    st.subheader("🔍 Image Processing")

    col1, col2, col3 = st.columns(3)

    processed = original.resize((224, 224))

    with col1:
        st.markdown("**Original Image**")
        st.image(
            original,
            use_container_width=True
        )

    with col2:
        st.markdown("**CNN Input (224 × 224)**")
        st.image(
            processed,
            use_container_width=True
        )

    with col3:
        st.markdown("**Preprocessing**")

        st.info("""
        RGB Image

        ↓

        Resize 224 × 224

        ↓

        Normalize 0–1

        ↓

        CNN Model
        """)

    image_array = np.array(
        processed,
        dtype=np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    with st.spinner("🧠 CNN is analyzing the image..."):

        prediction = model.predict(
            image_array,
            verbose=0
        )[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index]
    ) * 100

    st.markdown("---")
    st.subheader("🎯 AI Prediction")

    result_col, gauge_col = st.columns(2)

    with result_col:

        st.markdown(
            f"""
            <div class="result">

            <h3>Predicted Class</h3>

            <h1>
            {predicted_class.upper()}
            </h1>

            <p>
            {class_labels[predicted_class]}
            </p>

            <h3>
            Confidence: {confidence:.2f}%
            </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

    with gauge_col:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={
                    "text": "Model Confidence"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "bar": {
                        "thickness": 0.35
                    },
                    "steps": [
                        {
                            "range": [0, 50]
                        },
                        {
                            "range": [50, 75]
                        },
                        {
                            "range": [75, 100]
                        }
                    ]
                }
            )
        )

        fig.update_layout(
            height=300,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader(
        "📊 Seven-Class Probability Distribution"
    )

    probabilities = prediction * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[
                x.upper()
                for x in class_names
            ],
            y=probabilities,
            text=[
                f"{x:.2f}%"
                for x in probabilities
            ],
            textposition="auto"
        )
    )

    fig.update_layout(
        xaxis_title="Lesion Class",
        yaxis_title="Probability (%)",
        yaxis=dict(
            range=[0, 100]
        ),
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("🏆 Top 3 Predictions")

    sorted_indices = np.argsort(
        prediction
    )[::-1]

    for rank, index in enumerate(
        sorted_indices[:3],
        start=1
    ):

        name = class_names[index].upper()

        score = (
            float(prediction[index])
            * 100
        )

        st.write(
            f"**{rank}. {name} — "
            f"{score:.2f}%**"
        )

        st.progress(
            min(score / 100, 1.0)
        )

    st.markdown("---")

    st.subheader(
        "🧠 How the CNN Makes a Prediction"
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(
            "### 1️⃣\n**Input**\n\nSkin image"
        )

    with b:
        st.markdown(
            "### 2️⃣\n**Features**\n\nCNN extracts visual patterns"
        )

    with c:
        st.markdown(
            "### 3️⃣\n**Analysis**\n\nModel calculates probabilities"
        )

    with d:
        st.markdown(
            "### 4️⃣\n**Output**\n\nSeven-class prediction"
        )

st.markdown("---")

st.warning(
    "⚠️ Research Prototype: This AI classifier is designed "
    "for academic demonstration and research purposes. "
    "Predictions should not be considered a medical diagnosis "
    "or a substitute for professional medical advice."
)

st.markdown("""
<div class="footer">

<b>SkinGuard AI</b><br>

CNN-Based Skin Lesion Classifier

</div>
""", unsafe_allow_html=True)
