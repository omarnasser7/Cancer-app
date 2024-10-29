import numpy as np
import tensorflow as tf
from PIL import Image
import datetime
import os
import cv2

# Model loading function with error handling
def load_model(model_path):
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        return None
    
def preprocessing(img, img_shape):
    
    # Convert the PIL image to a NumPy array and then to BGR format
    img = np.array(img)  # Convert to NumPy array (RGB format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    # Convert PIL Image to NumPy array
    img = cv2.resize(img, img_shape)  # Resize to model's expected input size
    # Normalize if necessary (uncomment if needed)
    # img = img / 255.0  # Uncomment if normalization is needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_colon_cancer(image):
    model = load_model('colon_model.h5')
    if model is None:
        return "Error: Model could not be loaded"
    
    processed_img = preprocessing(image, img_shape=(256, 256))
    prediction = model.predict(processed_img)
    prediction = np.argmax(prediction, axis=1)[0]
    class_names = ['Colon adenocarcinoma', 'Colon benign tissue']
    prediction = class_names[prediction]
        
    result = {
        "Prediction": prediction,
        "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Note": "This is an automated analysis and should be verified by a healthcare professional."
    }
    
    return result

# Read the image using PIL
img_path = "imags/colon_n.jpeg"  # Make sure the path is correct
img = Image.open(img_path)

# Predict colon cancer
print(predict_colon_cancer(img))


# import numpy as np
# from PIL import Image
# import cv2
# import datetime
# import tensorflow as tf

# # Model loading function with error handling
# def load_model(model_path):
#     try:
#         return tf.keras.models.load_model(model_path)
#     except Exception as e:
#         print(f"Error loading model '{model_path}': {e}")
#         return None
    
# def preprocessing(img, img_shape):
#     # Resize to model's expected input size
#     img = cv2.resize(img, img_shape)  # Resize to model's expected input size
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img


# def predict_colon_cancer(image):
#     model = load_model('colon_model.h5')
#     if model is None:
#         return "Error: Model could not be loaded"
    
#     processed_img = preprocessing(image, img_shape=(256, 256))
#     prediction = model.predict(processed_img)
#     prediction = np.argmax(prediction, axis=1)[0]
#     class_names = ['Colon adenocarcinoma', 'Colon benign tissue']
#     prediction = class_names[prediction]
        
#     result = {
#         "Prediction": prediction,
#         "Analysis Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         "Note": "This is an automated analysis and should be verified by a healthcare professional."
#     }
    
#     return result

# # Read the image using PIL
# image_path = "imags/colon_n.jpeg"
# pil_image = Image.open(image_path)

# # Convert the PIL image to a NumPy array and then to BGR format
# rgb_image = np.array(pil_image)  # Convert to NumPy array (RGB format)
# bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

# # Make the prediction
# print(predict_colon_cancer(bgr_image))
