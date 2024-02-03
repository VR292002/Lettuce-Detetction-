import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt




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
    
def predict_with_yolo(model, image_path):
    with Image.open(image_path) as img:
        img = img.resize((255, 255))

        results = model(img, show=True)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        prediction = names_dict[probs.index(max(probs))]
        return prediction
    
    

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
























def evaluate_model(model, test_data):
    # Assuming test_data is a list of (image_path, true_labels) tuples
    all_predictions = []
    all_true_labels = []
    for image_path, true_labels in test_data:
        predictions = predict_with_yolo(model, image_path)
        all_predictions.append(predictions)
        all_true_labels.append(true_labels)

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    precision = precision_score(all_true_labels, all_predictions, average='weighted')
    recall = recall_score(all_true_labels, all_predictions, average='weighted')
    return accuracy, f1, precision, recall

# Assuming you have your test_data loaded
test_data_model1 = [("E:\Training-Data\graph\ib.jpg","Iceberg"),
                    ("E:\Training-Data\graph\lr.jpg","Lollo Rosso"),
                    ("E:\Training-Data\graph/rm.jpg","Romaine"),
                    ("E:\Training-Data\graph/rm2.jpg","Romaine")]
test_data_model2 = [("E:\Training-Data\graph/als.jpg","Alternaria Leaf Spot"),
                    ("E:\Training-Data\graph/atc.jpg","Anthracnose"),
                    ("E:\Training-Data\graph/rst.jpg","Rust"),
                    ("E:\Training-Data\graph/ap.jpg","Septoria Leaf Spot")]
test_data_model3 = [("E:\Training-Data\graph/ap.jpg","Aphids"),
                    ("E:\Training-Data\graph/aw.jpg","Armyworms"),
                    ("E:\Training-Data\graph/sb.jpg","Stinkbugs"),
                    ("E:\Training-Data\graph/lm.jpg","Leafminers")]



# Evaluate each model with its respective test data
model1_metrics = evaluate_model(model1, test_data_model1)
model2_metrics = evaluate_model(model2, test_data_model2)
model3_metrics = evaluate_model(model3, test_data_model3)


# Display model metrics
st.header("Model Performance Metrics")
st.subheader("Type")
st.write("Accuracy:", model1_metrics[0])
st.write("F1 Score:", model1_metrics[1])
st.write("Precision:", model1_metrics[2])
st.write("Recall:", model1_metrics[3])

st.header("Model Performance Metrics")
st.subheader("Disease")
st.write("Accuracy:", model2_metrics[0])
st.write("F1 Score:", model2_metrics[1])
st.write("Precision:", model2_metrics[2])
st.write("Recall:", model2_metrics[3])

st.header("Model Performance Metrics")
st.subheader("Pest")
st.write("Accuracy:", model3_metrics[0])
st.write("F1 Score:", model3_metrics[1])
st.write("Precision:", model3_metrics[2])
st.write("Recall:", model3_metrics[3])











# Create F1 score graph
fig, ax = plt.subplots()
models = ['Model 1', 'Model 2', 'Model 3']
f1_scores = [model1_metrics[1], model2_metrics[1], model3_metrics[1]]
ax.bar(models, f1_scores)
ax.set_title("F1 Scores for Each Model")
ax.set_xlabel("Model")
ax.set_ylabel("F1 Score")
st.pyplot(fig)