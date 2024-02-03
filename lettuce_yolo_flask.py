from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import numpy as np  # Import NumPy for array manipulation
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Load the YOLO model
model_path = "E:/Aquafoundry/runs/classify/train/weights/best.pt"
model = YOLO(model_path)

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            response = jsonify({'error': 'No file part'})
            print(response.data.decode('utf-8'))
            return response, 400

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            response = jsonify({'error': 'No selected file'})
            print(response.data.decode('utf-8'))
            return response, 400

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Save the file with a secure filename
            filename = secure_filename(file.filename)
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Run YOLO prediction on the uploaded image
            results = model(file_path)

            # Extract names and probabilities
            names_dict = results[0].names

            # Manually build a list from the tensor elements
            probs = [float(prob) for prob in results[0].probs.cpu().numpy()]

            # Print the JSON response
            response = jsonify({'names_dict': names_dict, 'probs': probs})
            print(response.data.decode('utf-8'))

            # Remove the uploaded file
            os.remove(file_path)

            return response

        else:
            response = jsonify({'error': 'Invalid file extension'})
            print(response.data.decode('utf-8'))
            return response, 400

    except Exception as e:
        response = jsonify({'error': str(e)})
        print(response.data.decode('utf-8'))
        return response, 500

if __name__ == '__main__':
    # Create the 'uploads' directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=False)
