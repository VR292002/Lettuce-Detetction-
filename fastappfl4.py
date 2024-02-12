from flask import Flask, render_template, send_from_directory
import pandas as pd
import requests

app = Flask(__name__)

@app.route('/')
def index():
    # Fetch JSON data from the FastAPI server
    response = requests.get("http://localhost:8000/")
    if response.status_code == 200:
        data = response.json()
        result = []

        # Iterate through each dictionary in the list
        for item in data:
            # Extract 'Vedaant' dictionary
            vedaant = item['Vedaant']

            # Extract 'image' and 'timestamp' values
            image_path = vedaant['image']
            timestamp = vedaant['timestamp']

            # Extract predictions dictionary
            predictions = vedaant['predictions']

            # Ensure the order is type, disease, pest
            for pred_type in ['type', 'disease', 'pest']:
                if pred_type in predictions:
                    # Flatten the nested list and create separate rows for each entry
                    for entry in predictions[pred_type]:
                        result.append({
                            'image': image_path,
                            'timestamp': timestamp,
                            'type': pred_type,
                            'category': entry[0],
                            'probability': entry[1]
                        })

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(result)

        # Add a new column 'photos' with full-size image HTML tags
        df['photos'] = df['image'].apply(lambda x: f'<img src="/image/{x}" style="max-width:none; max-height:none;"/>')

        # Rearrange columns such that 'timestamp' appears last
        df = df[['image', 'type', 'category', 'probability', 'timestamp', 'photos']]

        # Render the HTML template with the dataframe
        return render_template('indexfastappfl3.html', df=df.to_html(index=False, escape=False, render_links=True))
    else:
        return "Failed to fetch data from FastAPI server"

from PIL import Image
from io import BytesIO
from flask import send_file

@app.route('/image/<path:filename>')
def serve_image_resized(filename):
    directory, filename = filename.rsplit('/', 1)
    with Image.open(directory + '/' + filename) as img:
        img.thumbnail((100, 100))  # Resize the image to fit within a 100x100 box
        output = BytesIO()
        img.save(output, format='JPEG')  # Change the format as needed
        output.seek(0)
        return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)