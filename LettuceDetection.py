import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

dataset_path = 'E:/Training-Data/train'
num_classes = 13
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=10)

model.save('trained_model.h5')

def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array



import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('trained_model.h5')

# Define the prediction function
def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_lettuce_type(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0, class_index]

    if confidence >= 0.1:  # Adjust confidence threshold as needed
        return ["Alternaria Leaf Spot", "Angular Leaf Spot", "Anthracnose", "Bacterial Leaf Spot", "Black Rot Leaf Spot", "Cercospora Leaf", "Downy Mildew", "Gummy Stem Blight", "Healthy", "Mosaic Virus", "Powdery Mildew", "Rust", "Septoria Leaf Spot"][class_index]
    else:
        return "Unknown"

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
        predicted_class = predict_lettuce_type(img_path)

    st.success(f"Predicted class: {predicted_class}")

    # Optional: Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)

