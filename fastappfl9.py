from flask import Flask, render_template, request, send_file, send_from_directory
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)

def preprocess_data(data):
    result = []

    for item in data:
        vedaant = item['Vedaant']
        timestamp = vedaant['timestamp']
        predictions = vedaant['predictions']

        for pred_type in ['type', 'disease', 'pest']:
            if pred_type in predictions:
                for entry in predictions[pred_type]:
                    result.append({
                        'image': vedaant['image'],
                        'timestamp': timestamp,
                        'type': pred_type,
                        'category': entry[0],
                        'probability': entry[1]
                    })

    df = pd.DataFrame(result)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def generate_photos_column(df):
    df['photos'] = df['image'].apply(lambda x: f'<a href="#" onclick="showImage(\'{x}\')"><img src="/thumbnail/{x}" style="max-width:none; max-height:none;"/></a>')
    return df

@app.route('/')
def index():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Fetch JSON data from the FastAPI server
    response = requests.get("http://localhost:8000/")
    if response.status_code == 200:
        data = response.json()
        df = preprocess_data(data)

        # Filter the DataFrame based on the specified date range
        if start_date and end_date:
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()
            df = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]

        df = generate_photos_column(df)

        # Render the HTML template with the dataframe
        return render_template('indexfastappfl9.html', df=df.to_html(index=False, escape=False, render_links=True),
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
