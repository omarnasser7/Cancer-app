import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import datetime
import base64

# Convert an image file to a base64 string
with open("back2.jpg", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode("utf-8")

# Function to set the background image using base64
def add_bg_from_base64(b64_string):
    bg_base64 = f"data:image/png;base64,{b64_string}"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background at the start of the app
add_bg_from_base64(b64_string)

# Model loading function with error handling
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        return None

# Image preprocessing function
def preprocessing(img, img_shape=(224, 224)):
    img = np.array(img)
    img = cv2.resize(img, img_shape)  # Resize to model's expected input size
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function for leukemia
def predict_leukemia(image):
    model = load_model('leukemia_model.h5')
    if model is None:
        return {"Prediction": "Error: Model could not be loaded"}
    
    processed_img = preprocessing(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
    prediction = class_names[predicted_class]
        
    result = {
        "Prediction": prediction,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

# Streamlit UI
st.title("Leukemia Prediction App")
st.write("Upload a blood cell image to predict the likelihood of leukemia.")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            result = predict_leukemia(image)
        
        # Display results
        if "Error" in result["Prediction"]:
            st.error(result["Prediction"])
        else:
            st.success(f"Prediction: {result['Prediction']}")
            st.write(f"**Analysis Date**: {result['Analysis Date']}")
            st.info(result["Note"])

