from flask import Flask, render_template, request, send_file, send_from_directory
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

@app.route('/')
def index():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Fetch JSON data from the FastAPI server
    response = requests.get("http://localhost:8000/")
    if response.status_code == 200:
        data = response.json()
        result = []

        # Iterate through each dictionary in the list
        for item in data:
            # Extract 'Vedaant' dictionary
            vedaant = item['Vedaant']

            # Extract 'timestamp' value
            timestamp = vedaant['timestamp']

            # Extract predictions dictionary
            predictions = vedaant['predictions']

            # Ensure the order is type, disease, pest
            for pred_type in ['type', 'disease', 'pest']:
                if pred_type in predictions:
                    # Flatten the nested list and create separate rows for each entry
                    for entry in predictions[pred_type]:
                        result.append({
                            'image': vedaant['image'],
                            'timestamp': timestamp,
                            'type': pred_type,
                            'category': entry[0],
                            'probability': entry[1]
                        })

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(result)

        # Convert 'timestamp' column to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter the DataFrame based on the specified date range
        if start_date and end_date:
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()
            df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

        # Add a new column 'photos' with full-size image HTML tags
        df['photos'] = df['image'].apply(lambda x: f'<a href="/image/{x}" data-lightbox="photos"><img src="/thumbnail/{x}" style="max-width:none; max-height:none;"/></a>')

        # Rearrange columns such that 'timestamp' appears last
        df = df[['image', 'type', 'category', 'probability', 'timestamp', 'photos']]

        # Render the HTML template with the dataframe
        return render_template('indexfastappfl7.html', df=df.to_html(index=False, escape=False, render_links=True),
                               start_date=start_date, end_date=end_date)
    else:
        return "Failed to fetch data from FastAPI server"

@app.route('/image/<path:filename>')
def serve_image(filename):
    directory, filename = os.path.split(filename)
    return send_from_directory(directory, filename)

@app.route('/thumbnail/<path:filename>')
def serve_thumbnail(filename):
    directory, filename = os.path.split(filename)
    with Image.open(os.path.join(directory, filename)) as img:
        img.thumbnail((100, 100))  # Resize the image to fit within a 100x100 box
        output = BytesIO()
        img.save(output, format='JPEG')  # Change the format as needed
        output.seek(0)
        return send_file(output, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)