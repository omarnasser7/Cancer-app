# import gradio as gr
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import cv2
# import datetime
# import base64
# from colorama import Fore, Style, init

# # Initialize colorama for terminal output (if needed)
# init(autoreset=True)

# # Function to load model with error handling
# def load_model(model_path):
#     try:
#         return tf.keras.models.load_model(model_path)
#     except Exception as e:
#         print(f"Error loading model '{model_path}': {e}")
#         return None

# # Function to resize image if it's larger than a specific size
# def resize_image_if_needed(img, max_size=(256, 256)):
#     if img.shape[0] > max_size[0] or img.shape[1] > max_size[1]:
#         return cv2.resize(img, max_size)
#     return img

# # Preprocessing functions
# def skin_preprocess(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = resize_image_if_needed(img, (224, 224))
#     img = img / 255
#     img = np.expand_dims(img, axis=0)
#     return img

# def custom_preprocessing(img, img_shape):
#     img = np.array(img)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     img = resize_image_if_needed(img, img_shape)
#     img = np.expand_dims(img, axis=0)
#     return img

# def preprocessing(img, img_shape):
#     img = np.array(img)
#     img = resize_image_if_needed(img, img_shape)
#     img = np.expand_dims(img, axis=0)
#     return img

# # Prediction functions for different cancer types
# def predict_skin_cancer(image):
#     model = load_model('skin_model.keras')
#     if model is None:
#         return {"Error": "Model could not be loaded"}
    
#     processed_img = skin_preprocess(image)
#     prediction = model.predict(processed_img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     class_names = ['Benign', 'Malignant']
#     prediction_label = class_names[predicted_class]
        
#     result = {
#         "Prediction": prediction_label,
#         "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Note": "This is an automated analysis and should be verified by a healthcare professional."
#     }
    
#     return result

# def predict_colon_cancer(image):
#     model = load_model('colon_model.h5')
#     if model is None:
#         return {"Error": "Model could not be loaded"}
    
#     processed_img = custom_preprocessing(image, (256, 256))
#     prediction = model.predict(processed_img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     class_names = ['Colon adenocarcinoma', 'Colon benign tissue']
#     prediction_label = class_names[predicted_class]
        
#     result = {
#         "Prediction": prediction_label,
#         "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Note": "This is an automated analysis and should be verified by a healthcare professional."
#     }
    
#     return result

# def predict_leukemia(image):
#     model = load_model('leukemia_model.h5')
#     if model is None:
#         return {"Error": "Model could not be loaded"}
    
#     processed_img = preprocessing(image, (224, 224))
#     prediction = model.predict(processed_img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     class_names = ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B']
#     prediction_label = class_names[predicted_class]
    
#     result = {
#         "Prediction": prediction_label,
#         "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Note": "This is an automated analysis and should be verified by a healthcare professional."
#     }
    
#     return result

# def predict_lung_cancer(image):
#     model = load_model('lung_model.h5')
#     if model is None:
#         return {"Error": "Model could not be loaded"}
    
#     processed_img = custom_preprocessing(image, (256, 256))
#     prediction = model.predict(processed_img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     class_names = ['Lung adenocarcinoma', 'Lung benign tissue', 'Lung squamous cell carcinoma']
#     prediction_label = class_names[predicted_class]
        
#     result = {
#         "Prediction": prediction_label,
#         "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Note": "This is an automated analysis and should be verified by a healthcare professional."
#     }
    
#     return result

# # Function to encode an image in base64
# def encode_image_base64(image_path):
#     with open(image_path, "rb") as img_file:
#         return base64.b64encode(img_file.read()).decode('utf-8')

# # Encode the background image
# background_image_base64 = encode_image_base64("back2.jpg")

# # Create the Gradio interface
# def create_cancer_detection_interface():
#     # CSS styling with base64 background image
#     css = f"""
#     .gradio-container {{
#         font-family: 'Arial', sans-serif;
#         background-image: url('data:image/jpg;base64,{background_image_base64}');  
#         background-size: cover;
#         background-position: center;
#         color: white;
#     }}
#     .gr-tab, .gr-button, .gr-image, .gr-json {{
#         background-color: rgba(0, 0, 0, 0.6);
#         border-radius: 8px;
#     }}
#     .prediction-label {{
#         font-size: 1.2em;
#         font-weight: bold;
#         margin-top: 1em;
#     }}
#     """
    
#     # Create tabs for different cancer types
#     with gr.Blocks(css=css) as demo:
#         gr.Markdown("# Cancer Detection App")
#         gr.Markdown("Upload an image for analysis. Please consult healthcare professionals for accurate diagnosis.")
        
#         with gr.Tabs():
#             with gr.Tab("Skin Cancer"):
#                 with gr.Row():
#                     with gr.Column():
#                         skin_input = gr.Image(label="Upload Skin Image")
#                         skin_button = gr.Button("Analyze Skin Image")
#                     with gr.Column():
#                         skin_output = gr.JSON(label="Analysis Results")
#                 skin_button.click(
#                     predict_skin_cancer,
#                     inputs=skin_input,
#                     outputs=skin_output
#                 )
            
#             with gr.Tab("Colon Cancer"):
#                 with gr.Row():
#                     with gr.Column():
#                         colon_input = gr.Image(label="Upload Colon Image")
#                         colon_button = gr.Button("Analyze Colon Image")
#                     with gr.Column():
#                         colon_output = gr.JSON(label="Analysis Results")
#                 colon_button.click(
#                     predict_colon_cancer,
#                     inputs=colon_input,
#                     outputs=colon_output
#                 )
            
#             with gr.Tab("Leukemia"):
#                 with gr.Row():
#                     with gr.Column():
#                         leukemia_input = gr.Image(label="Upload Blood Sample Image")
#                         leukemia_button = gr.Button("Analyze Blood Sample")
#                     with gr.Column():
#                         leukemia_output = gr.JSON(label="Analysis Results")
#                 leukemia_button.click(
#                     predict_leukemia,
#                     inputs=leukemia_input,
#                     outputs=leukemia_output
#                 )
            
#             with gr.Tab("Lung Cancer"):
#                 with gr.Row():
#                     with gr.Column():
#                         lung_input = gr.Image(label="Upload Lung Image")
#                         lung_button = gr.Button("Analyze Lung Image")
#                     with gr.Column():
#                         lung_output = gr.JSON(label="Analysis Results")
#                 lung_button.click(
#                     predict_lung_cancer,
#                     inputs=lung_input,
#                     outputs=lung_output
#                 )
        
#         gr.Markdown("""### Important Notes:
#         - This is an automated analysis tool and should not be used as a substitute for professional medical diagnosis.
#         - Please consult with healthcare professionals for accurate diagnosis and treatment options.
#         - Save the analysis results for your records and discuss them with your doctor.
#         """)
    
#     return demo

# # Launch the app
# if __name__ == "__main__":
#     demo = create_cancer_detection_interface()
#     demo.launch()

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import datetime

# Function to load model with error handling
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None

# Image preprocessing functions
def preprocess_image(img, target_size=(224, 224)):
    if img is None:
        return None
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction functions for different cancer types
def predict_cancer(image, cancer_type):
    if image is None:
        return {"Error": "No image provided"}
    
    # Model paths for different cancer types
    model_paths = {
        "skin": "skin_model.keras",
        "colon": "colon_model.h5",
        "leukemia": "leukemia_model.h5",
        "lung": "lung_model.h5"
    }
    
    # Class names for different cancer types
    class_names = {
        "skin": ['Benign', 'Malignant'],
        "colon": ['Colon adenocarcinoma', 'Colon benign tissue'],
        "leukemia": ['Benign', 'Malignant_Pre-B', 'Malignant_Pro-B', 'Malignant_early Pre-B'],
        "lung": ['Lung adenocarcinoma', 'Lung benign tissue', 'Lung squamous cell carcinoma']
    }
    
    # Load and preprocess image
    processed_img = preprocess_image(image)
    if processed_img is None:
        return {"Error": "Image preprocessing failed"}
    
    # Load model
    model = load_model(model_paths[cancer_type])
    if model is None:
        return {"Error": "Model loading failed"}
    
    # Make prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(prediction[0][predicted_class] * 100)
    
    result = {
        "Prediction": class_names[cancer_type][predicted_class],
        "Confidence": f"{confidence:.2f}%",
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

# Custom CSS for better UI
custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #1a365d 0%, #2d3748 100%);
        color: white;
    }
    .tabs {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .tabitem {
        background-color: rgba(255, 255, 255, 0.05);
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        color: white;
        transition: all 0.3s ease;
    }
    .tabitem.selected {
        background-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    }
    .panel {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
    }
    .image-preview {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
    }
    .analyze-btn {
        background-color: #4299e1;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .analyze-btn:hover {
        background-color: #3182ce;
    }
    .warning {
        background-color: rgba(255, 87, 34, 0.1);
        border-left: 4px solid #ff5722;
        padding: 15px;
        margin-top: 20px;
        border-radius: 4px;
    }
"""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(
            """
            # Cancer Detection Assistant
            Upload medical images for AI-powered analysis. Please note that this tool is for preliminary screening only.
            """
        )
        
        with gr.Tabs() as tabs:
            for cancer_type in ["skin", "colon", "leukemia", "lung"]:
                with gr.Tab(f"{cancer_type.title()} Cancer"):
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(
                                label="Upload Image",
                                type="numpy",
                                height=300
                            )
                            analyze_btn = gr.Button(
                                "Analyze Image",
                                variant="primary"
                            )
                        
                        with gr.Column():
                            output = gr.JSON(label="Analysis Results")
                    
                    analyze_btn.click(
                        fn=lambda img: predict_cancer(img, cancer_type),
                        inputs=image_input,
                        outputs=output
                    )
        
        gr.Markdown(
            """
            ### Important Notes:
            - This tool is for preliminary screening purposes only
            - Results should be verified by healthcare professionals
            - Image quality affects analysis accuracy
            - Regular medical check-ups are essential
            """
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )