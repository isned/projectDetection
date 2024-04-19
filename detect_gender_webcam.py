import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Load model
model = load_model('gender_detection.model')

# Labels for classes
classes = ['man', 'woman']

# Function to perform gender detection
def detect_gender(image):
    # Preprocess the image
    image = image.resize((96, 96))  # Resize image to match model input shape
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict gender
    prediction = model.predict(image)[0]
    gender_index = np.argmax(prediction)
    gender = classes[gender_index]
    confidence = prediction[gender_index] * 100

    return gender, confidence

# Streamlit UI
st.title("Gender Detection with Streamlit")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    image.thumbnail((300, 300))  # Resize the image to be maximum 300x300
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform gender detection when button is clicked
    if st.button("Detect Gender"):
        gender, confidence = detect_gender(image)
        st.write(f"Gender: {gender}, Confidence: {confidence:.2f}%")
