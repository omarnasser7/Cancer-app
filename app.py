import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64
import cv2
import datetime

# Streamlit page configuration
st.set_page_config(page_title="Cancer Detection App", layout="wide")

# Function to get base64 of binary file for background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set custom styles for the app
def set_custom_styles():
    st.markdown("""
        <style>
        .main-header {
            color: #2c3e50;
            text-align: center;
            padding: 20px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .button-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .stButton>button {
            width: 200px;
            height: 60px;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

# Enhanced display_result function
def display_result(prediction):
    # Create a container for the result
    result_container = st.container()
    
    with result_container:
        # Split into columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Style definitions
            result_styles = """
                <style>
                    .result-box {
                        padding: 20px;
                        border-radius: 10px;
                        background-color: rgba(255, 255, 255, 0.9);
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin: 20px 0;
                        text-align: center;
                    }
                    .result-header {
                        color: #2c3e50;
                        font-size: 24px;
                        font-weight: bold;
                        margin-bottom: 15px;
                    }
                    .result-value {
                        font-size: 28px;
                        font-weight: bold;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 10px 0;
                    }
                    .result-info {
                        color: #34495e;
                        font-size: 16px;
                        margin-top: 15px;
                        line-height: 1.5;
                    }
                    .warning-text {
                        color: #e74c3c;
                        font-size: 14px;
                        margin-top: 20px;
                        font-style: italic;
                    }
                </style>
            """
            
            # Determine result color and message based on prediction
            def get_result_styling(prediction):
                if any(word in prediction.lower() for word in ['benign', 'normal']):
                    return {
                        'color': '#27ae60',
                        'message': 'The analysis suggests normal/benign tissue. However, please consult with a healthcare professional for proper diagnosis.'
                    }
                else:
                    return {
                        'color': '#e74c3c',
                        'message': 'The analysis suggests potentially concerning results. Please consult with a healthcare professional immediately for proper diagnosis and treatment options.'
                    }
            
            # Get styling based on prediction
            styling = get_result_styling(prediction)
            
            # Display the result
            st.markdown(result_styles, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="result-box">
                    <div class="result-header">Analysis Result</div>
                    <div class="result-value" style="background-color: {styling['color']}; color: white;">
                        {prediction}
                    </div>
                    <div class="result-info">
                        {styling['message']}
                    </div>
                    <div class="warning-text">
                        This is an automated analysis tool and should not be used as a substitute for professional medical diagnosis.
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Add additional information based on the type of cancer
            if prediction != "":
                st.markdown("---")
                st.markdown("### Next Steps")
                st.markdown("""
                1. Save these results for your records
                2. Schedule an appointment with your healthcare provider
                3. Prepare any questions you may have about the results
                """)

# Function to save results
def save_results(prediction, image):
    if st.button("Download Results"):
        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a PDF report
        try:
            # Create a temporary file to save results
            result_filename = f"cancer_detection_results_{timestamp}.txt"
            
            with open(result_filename, "w") as f:
                f.write(f"Cancer Detection Analysis Results\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Prediction: {prediction}\n")
                f.write(f"\nNote: This is an automated analysis and should be verified by a healthcare professional.")
            
            # Create download button
            with open(result_filename, "rb") as f:
                st.download_button(
                    label="Download Results Report",
                    data=f,
                    file_name=result_filename,
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Error saving results: {e}")

# Function to load image
def upload_image():
    st.markdown("""
        <style>
        .upload-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image.", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            return img
        else:
            st.write("Please upload an image.")
            st.markdown('</div>', unsafe_allow_html=True)
            return None

# Define the home page layout
def main():
    set_custom_styles()
    
    # Set the background image
    try:
        set_background('back1.jpg')
    except Exception as e:
        st.warning("Background image not found. Using default background.")
    
    st.markdown('<h1 class="main-header">Cancer Detection App</h1>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Choose a type of cancer to detect:</h3>", unsafe_allow_html=True)
    
    # Create columns for buttons with better styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Skin Cancer", key="skin"):
            skin_cancer_page()
    with col2:
        if st.button("Colon Cancer", key="colon"):
            colon_cancer_page()
    with col3:
        if st.button("Leukemia", key="leukemia"):
            leukemia_page()
    with col4:
        if st.button("Lung Cancer", key="lung"):
            lung_cancer_page()

# Cancer detection pages
def skin_cancer_page():
    st.title("Skin Cancer Detection")
    model = load_model('skin_model.keras')
    if model:
        img = upload_image()
        if img is not None:
            with st.spinner('Processing image...'):
                processed_img = preprocess_skin_cancer(img)
                prediction = skin_prediction(model, processed_img)
                display_result(prediction)
                save_results(prediction, img)

def colon_cancer_page():
    st.title("Colon Cancer Detection")
    model = load_model('colon_model.h5')
    if model:
        img = upload_image()
        if img is not None:
            with st.spinner('Processing image...'):
                processed_img = preprocess_colon_cancer(img)
                prediction = colon_prediction(model, processed_img)
                display_result(prediction)
                save_results(prediction, img)

def leukemia_page():
    st.title("Leukemia Detection")
    model = load_model('leukemia_model.h5')
    if model:
        img = upload_image()
        if img is not None:
            with st.spinner('Processing image...'):
                processed_img = preprocess_leukemia(img)
                prediction = leukemia_prediction(model, processed_img)
                display_result(prediction)
                save_results(prediction, img)

def lung_cancer_page():
    st.title("Lung Cancer Detection")
    model = load_model('lung_model.h5')
    if model:
        img = upload_image()
        if img is not None:
            with st.spinner('Processing image...'):
                processed_img = preprocess_lung_cancer(img)
                prediction = lung_prediction(model, processed_img)
                display_result(prediction)
                save_results(prediction, img)

# Model loading function with error handling
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success(f"Model '{model_path}' loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model '{model_path}': {e}")
        return None

# Dummy preprocess and prediction functions
def preprocess_skin_cancer(image):
    # Add your preprocessing logic here
    return np.array(image.resize((224, 224)))  # Example resize, adjust as needed

def skin_prediction(model, image):
    # Add your prediction logic here
    return "Benign"  # Placeholder return value, replace with actual prediction logic

def preprocess_colon_cancer(image):
    return np.array(image.resize((224, 224)))

def colon_prediction(model, image):
    return "Malignant"  # Placeholder

def preprocess_leukemia(image):
    return np.array(image.resize((224, 224)))

def leukemia_prediction(model, image):
    return "Normal"  # Placeholder

def preprocess_lung_cancer(image):
    return np.array(image.resize((224, 224)))

def lung_prediction(model, image):
    return "Malignant"  # Placeholder

if __name__ == "__main__":
    main()
