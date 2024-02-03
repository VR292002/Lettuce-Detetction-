import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model1.h5')
model2 = load_model('trained_model2.h5')
model3 = load_model('trained_model3.h5')

# Define the prediction function
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_disease_type(model,img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0, class_index]

    if confidence >= 0.3:  # Adjust confidence threshold as needed
        return ["Alternaria Leaf Spot", "Angular Leaf Spot", "Anthracnose", "Bacterial Leaf Spot", "Black Rot Leaf Spot", "Cercospora Leaf", "Downy Mildew", "Gummy Stem Blight", "Healthy", "Mosaic Virus", "Powdery Mildew", "Rust", "Septoria Leaf Spot"][class_index]
    else:
        return "Unknown"
   
def predict_top_let_classes(model, img_path, top_k=3):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    top_classes = np.argsort(predictions[0])[::-1][:top_k]
    top_classes_labels = ["Butterhead", "Green Batavia", "Iceberg", "Lollo Rosso", "Romaine"]
    top_classes_names = [top_classes_labels[i] for i in top_classes]
    top_confidences = predictions[0, top_classes]
    return list(zip(top_classes_names, top_confidences))


def predict_top_dis_classes(model, img_path, top_k=3):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    top_classes = np.argsort(predictions[0])[::-1][:top_k]
    top_classes_labels = ["Alternaria Leaf Spot", "Angular Leaf Spot", "Anthracnose", "Bacterial Leaf Spot", "Black Rot Leaf Spot", "Cercospora Leaf", "Downy Mildew", "Gummy Stem Blight", "Healthy", "Mosaic Virus", "Powdery Mildew", "Rust", "Septoria Leaf Spot"]
    top_classes_names = [top_classes_labels[i] for i in top_classes]
    top_confidences = predictions[0, top_classes]
    return list(zip(top_classes_names, top_confidences))


def predict_top_pes_classes(model, img_path, top_k=3):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    top_classes = np.argsort(predictions[0])[::-1][:top_k]
    top_classes_labels = ["Aphids", "Armyworms", "Cabbage Looper", "Colorado Potato Beetle", "Cutworms", "Flea Beetles", "Healthy", "Hornworms", "Leafminers", "Stinkbugs", "Wireworms"]
    top_classes_names = [top_classes_labels[i] for i in top_classes]
    top_confidences = predictions[0, top_classes]
    return list(zip(top_classes_names, top_confidences))

# Create the Streamlit app
st.title("Lettuce Image Classification")

uploaded_file = st.file_uploader("Choose an image of a lettuce leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Classifying..."):
        
        predicted_class_model1 = predict_top_dis_classes(model, img_path)
        predicted_class_model2 = predict_top_let_classes(model2, img_path)
        predicted_class_model3 = predict_top_pes_classes(model3, img_path)
    
    st.success("Top 3 Predicted Lettuces:")
    for class_name, confidence in predicted_class_model2:
        st.write(f"{class_name}: {confidence:.2%}")

    st.success("Top 3 Predicted Diseases:")
    for class_name, confidence in predicted_class_model1:
        st.write(f"{class_name}: {confidence:.2%}")

    st.success("Top 3 Predicted Pests:")
    for class_name, confidence in predicted_class_model3:
        st.write(f"{class_name}: {confidence:.2%}")

    # Optional: Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)