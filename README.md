Skin Lesion Classifier

A deep-learning project for classifying skin-lesion images using the HAM10000 dataset.

Project Overview

This project uses a trained TensorFlow/Keras model to classify uploaded skin-lesion images into different categories.

Features

- Upload a skin-lesion image
- Process the image using a deep-learning model
- Display the model's prediction
- Simple Streamlit web interface

Dataset

The project was developed using the HAM10000 (Human Against Machine with 10000 training images) dataset.

The original dataset is not included in this repository because of its large size.

Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- Streamlit
- GitHub

Project Files

- "app.py" — Streamlit application
- "requirements.txt" — Python dependencies
- "skin_lesion_model.keras" — trained model (stored separately because of its large file size)

Important Note

This project is intended for educational and research purposes only. Model predictions should not be considered a medical diagnosis.

How to Run

Install the required packages:

pip install -r requirements.txt

Then run:

streamlit run app.py
