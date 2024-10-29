import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import datetime
import base64
from colorama import Fore, Style, init

# Initialize colorama for terminal output (if needed)
init(autoreset=True)

# Function to load model with error handling
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None

# Function to resize image if it's larger than a specific size
def resize_image_if_needed(img, max_size=(256, 256)):
    if img.shape[0] > max_size[0] or img.shape[1] > max_size[1]:
        return cv2.resize(img, max_size)
    return img

# Preprocessing functions
def skin_preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = resize_image_if_needed(img, (224, 224))
    img = img / 255
    img = np.expand_dims(img, axis=0)
    return img

def custom_preprocessing(img, img_shape):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = resize_image_if_needed(img, img_shape)
    img = np.expand_dims(img, axis=0)
    return img

def preprocessing(img, img_shape):
    img = np.array(img)
    img = resize_image_if_needed(img, img_shape)
    img = np.expand_dims(img, axis=0)
    return img

# Prediction functions for different cancer types
def predict_skin_cancer(image):
    model = load_model('skin_model.keras')
    if model is None:
        return {"Error": "Model could not be loaded"}
    
    processed_img = skin_preprocess(image)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Benign', 'Malignant']
    prediction_label = class_names[predicted_class]
        
    result = {
        "Prediction": prediction_label,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

def predict_colon_cancer(image):
    model = load_model('colon_model.h5')
    if model is None:
        return {"Error": "Model could not be loaded"}
    
    processed_img = custom_preprocessing(image, (256, 256))
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Colon adenocarcinoma', 'Colon benign tissue']
    prediction_label = class_names[predicted_class]
        
    result = {
        "Prediction": prediction_label,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

def predict_leukemia(image):
    model = load_model('leukemia_model.h5')
    if model is None:
        return {"Error": "Model could not be loaded"}
    
    processed_img = preprocessing(image, (224, 224))
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
    prediction_label = class_names[predicted_class]
    
    result = {
        "Prediction": prediction_label,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

def predict_lung_cancer(image):
    model = load_model('lung_model.h5')
    if model is None:
        return {"Error": "Model could not be loaded"}
    
    processed_img = custom_preprocessing(image, (256, 256))
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['Lung adenocarcinoma', 'Lung benign tissue', 'Lung squamous cell carcinoma']
    prediction_label = class_names[predicted_class]
        
    result = {
        "Prediction": prediction_label,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

# Function to encode an image in base64
def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Encode the background image
background_image_base64 = encode_image_base64("back2.jpg")

# Create the Gradio interface
def create_cancer_detection_interface():
    # CSS styling with base64 background image
    css = f"""
    .gradio-container {{
        font-family: 'Arial', sans-serif;
        background-image: url('data:image/jpg;base64,{background_image_base64}');  
        background-size: cover;
        background-position: center;
        color: white;
    }}
    .gr-tab, .gr-button, .gr-image, .gr-json {{
        background-color: rgba(0, 0, 0, 0.6);
        border-radius: 8px;
    }}
    .prediction-label {{
        font-size: 1.2em;
        font-weight: bold;
        margin-top: 1em;
    }}
    """
    
    # Create tabs for different cancer types
    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Cancer Detection App")
        gr.Markdown("Upload an image for analysis. Please consult healthcare professionals for accurate diagnosis.")
        
        with gr.Tabs():
            with gr.Tab("Skin Cancer"):
                with gr.Row():
                    with gr.Column():
                        skin_input = gr.Image(label="Upload Skin Image")
                        skin_button = gr.Button("Analyze Skin Image")
                    with gr.Column():
                        skin_output = gr.JSON(label="Analysis Results")
                skin_button.click(
                    predict_skin_cancer,
                    inputs=skin_input,
                    outputs=skin_output
                )
            
            with gr.Tab("Colon Cancer"):
                with gr.Row():
                    with gr.Column():
                        colon_input = gr.Image(label="Upload Colon Image")
                        colon_button = gr.Button("Analyze Colon Image")
                    with gr.Column():
                        colon_output = gr.JSON(label="Analysis Results")
                colon_button.click(
                    predict_colon_cancer,
                    inputs=colon_input,
                    outputs=colon_output
                )
            
            with gr.Tab("Leukemia"):
                with gr.Row():
                    with gr.Column():
                        leukemia_input = gr.Image(label="Upload Blood Sample Image")
                        leukemia_button = gr.Button("Analyze Blood Sample")
                    with gr.Column():
                        leukemia_output = gr.JSON(label="Analysis Results")
                leukemia_button.click(
                    predict_leukemia,
                    inputs=leukemia_input,
                    outputs=leukemia_output
                )
            
            with gr.Tab("Lung Cancer"):
                with gr.Row():
                    with gr.Column():
                        lung_input = gr.Image(label="Upload Lung Image")
                        lung_button = gr.Button("Analyze Lung Image")
                    with gr.Column():
                        lung_output = gr.JSON(label="Analysis Results")
                lung_button.click(
                    predict_lung_cancer,
                    inputs=lung_input,
                    outputs=lung_output
                )
        
        gr.Markdown("""### Important Notes:
        - This is an automated analysis tool and should not be used as a substitute for professional medical diagnosis.
        - Please consult with healthcare professionals for accurate diagnosis and treatment options.
        - Save the analysis results for your records and discuss them with your doctor.
        """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_cancer_detection_interface()
    demo.launch()
