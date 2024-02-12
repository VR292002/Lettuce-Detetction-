from flask import Flask, render_template
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
            image = vedaant['image']
            timestamp = vedaant['timestamp']

            # Extract predictions dictionary
            predictions = vedaant['predictions']

            # Ensure the order is type, disease, pest
            for pred_type in ['type', 'disease', 'pest']:
                if pred_type in predictions:
                    # Flatten the nested list and create separate rows for each entry
                    for entry in predictions[pred_type]:
                        result.append({
                            'image': image,
                            'timestamp': timestamp,
                            'type': pred_type,
                            'category': entry[0],
                            'probability': entry[1]
                        })

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(result)

        # Rearrange columns such that 'timestamp' appears last
        df = df[['image', 'type', 'category', 'probability', 'timestamp']]

        return render_template('indexfastappfl.html', df=df.to_html(index=False))
    else:
        return "Failed to fetch data from FastAPI server"

if __name__ == '__main__':
    app.run(debug=True)
