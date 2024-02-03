# app_with_yolo.py
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

def predict_disease(image_path):
    # Load the YOLOv5 model
    model = YOLO("E:/Aquafoundry/runs/classify/train/weights/best.pt")

    # Read and resize the input image using Pillow (PIL)
    with Image.open(image_path) as img:
        img = img.resize((255, 255))

    # Perform prediction on the image
    results = model(img, show=True)

    # Extract relevant information from the prediction
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Determine the prediction, considering the "unknown" option
    if max(probs) >= 0.5:  # Adjust confidence threshold as needed
        prediction = names_dict[probs.index(max(probs))]
    else:
        prediction = "unknown"

    return prediction

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    file = request.files['image']

    # Save the file temporarily (optional)
    image_path = 'temp_image.jpg'
    file.save(image_path)

    # Pass the image to the YOLOv5 model for prediction
    prediction = predict_disease(image_path)

    # Render the template with the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=False)
