import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

# Load model
model = load_model('gender_detection.model')

# Streamlit UI
st.title("Gender Detection with Streamlit")

# Open webcam
webcam = cv2.VideoCapture(0)

# Labels for classes
classes = ['man','woman']

# Function to perform gender detection
def detect_gender(frame):
    # Apply face detection
    faces, confidences = cv.detect_face(frame)

    # Loop through detected faces
    for idx, face in enumerate(faces):
        startX, startY, endX, endY = face

        # Crop the detected face region
        face_crop = frame[startY:endY, startX:endX]

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on face
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        confidence = conf[idx] * 100

        # Draw rectangle and label on face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}%', (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Main streamlit app
while webcam.isOpened():
    status, frame = webcam.read()

    # Perform gender detection on the frame
    frame = detect_gender(frame)

    # Display the output frame
    st.image(frame, channels="BGR", use_column_width=True)

    # Check for keyboard input to stop the loop
    if st.button("Stop"):
        break

# Release resources
webcam.release()
