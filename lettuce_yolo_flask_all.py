# app_with_multiple_models.py
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

def load_yolo_model(path):
    return YOLO(path)

def predict_with_yolo(model, image_path):
    with Image.open(image_path) as img:
        img = img.resize((255, 255))

        results = model(img, show=True)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        prediction = "unknown" if max(probs) < 0.5 else names_dict[probs.index(max(probs))]
        return prediction
    
def predict_with_yolod(model, image_path):
    with Image.open(image_path) as img:
        img = img.resize((255, 255))

        results = model(img, show=True)
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()

        prediction = "unknown" if max(probs) < 0.2 else names_dict[probs.index(max(probs))]
        return prediction

@app.route('/')
def home():
    return render_template('index.html', predictions=None)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image_path = 'temp_image.jpg'
    file.save(image_path)

    # Load the models
    model1 = load_yolo_model("E:/Aquafoundry/runs/classify/train/weights/best.pt")
    model2 = load_yolo_model("E:\Aquafoundry/runs\classify/train9\weights/best.pt")
    model3 = load_yolo_model("E:\Aquafoundry/runs\classify/train8\weights/best.pt")

    # Make predictions with each model
    predictions = [
        predict_with_yolo(model1, image_path),
        predict_with_yolod(model2, image_path),
        predict_with_yolo(model3, image_path),
    ]

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=False)
