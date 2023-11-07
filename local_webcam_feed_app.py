import cv2
import streamlit as st

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import os
import tensorflow_probability as tfp

import keras
import keras.backend as K
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st
from keras.models import load_model

def resize_image(image, size):
    # Convert the image to PIL Image object
    pil_image = Image.fromarray(image.copy())
    # Resize the image
    resized_image = pil_image.resize(size)
    # Convert the resized image back to numpy array
    resized_array = np.array(resized_image)
    return resized_array


#Function to preprocess the input image
def preprocess_image(image):
    image = resize_image(image, (224, 224))
    image = img_to_array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)  # Add extra dimension to match model input shape
    return image

# Load the pre-trained model
model = load_model('/Users/preetikaparashar/Desktop/mlproject/project_model.h5')

# Function to make BMI prediction
def predict_bmi(image):
    preprocessed_image = preprocess_image(image)
    bmi_prediction = model.predict(preprocessed_image)
    #bmi_prediction = model.predict(image)
    return bmi_prediction[0][0]

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def draw_face_with_bmi(frame, x, y, w, h, bmi):
    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Write the BMI prediction on the box
    cv2.putText(frame, f'BMI: {bmi:.1f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def preprocess_image_2(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the image
        face_region = image[y:y+h, x:x+w]
        # Resize the face region to match the model input size
        resized_face = resize_image(face_region, (224, 224))
        # Perform BMI prediction using your model
        bmi = predict_bmi(resized_face)
        # Draw the face with BMI prediction
        draw_face_with_bmi(image, x, y, w, h, bmi)

    return image

def webcam_capture():
    # Create a VideoCapture object to access the webcam
    cap = cv2.VideoCapture(0)  # 0 represents the default webcam index
    while True:
        # Read frames from the webcam
        ret, frame = cap.read()
        # Preprocess the frame
        processed_frame = preprocess_image_2(frame)
        # Display the preprocessed frame
        cv2.imshow('Webcam', processed_frame)
        #st.image(processed_frame, channels="BGR")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start capturing webcam input with face frames
webcam_capture()
