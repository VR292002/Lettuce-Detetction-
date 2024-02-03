import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os



model1 = YOLO("E:/Aquafoundry/runs/classify/train/weights/best.pt")
model2 = YOLO("E:\Aquafoundry/runs\classify/train9\weights/best.pt")
model3 = YOLO("E:\Aquafoundry/runs\classify/train8\weights/best.pt")



def get_top_predictions(results):
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]  # Get indices of top 3 probabilities
    return [(names_dict[i], probs[i]) for i in indices]



def predict_with_model(model, image_path):
    with Image.open(image_path) as img:
        img = img.resize((255, 255))

        results = model(img, show=True)  # Disable image display for prediction
        top_predictions = get_top_predictions(results)
        return top_predictions
    

st.title("Lettuce Type, Disease and Pest Detection")
st.header("Please Upload an Image of a Lettuce")




uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    temp_dir = "temp_images"
    os.makedirs(temp_dir, exist_ok=True)
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())



    with st.spinner("Classifying..."):
            
            predicted_class_model1 = predict_with_model(model1, img_path)
            predicted_class_model2 = predict_with_model(model2, img_path)
            predicted_class_model3 = predict_with_model(model3, img_path)



    
    st.success("Top 3 Predicted Lettuces:")
    for class_name, confidence in predicted_class_model1:
        st.write(f"{class_name}: {confidence:.2%}")

    st.success("Top 3 Predicted Diseases:")
    for class_name, confidence in predicted_class_model2:
        st.write(f"{class_name}: {confidence:.2%}")

    st.success("Top 3 Predicted Pests:")
    for class_name, confidence in predicted_class_model3:
        st.write(f"{class_name}: {confidence:.2%}")

    # Optional: Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=300)


    

