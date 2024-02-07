from flask import Flask, render_template, request
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd

app = Flask(__name__)

# Connect to Couchbase
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant')
collection = bucket.default_collection()

# Function to query documents and extract data
def get_data():
    query = "SELECT * FROM Vedaant"
    rows = cluster.query(query)
    data = []
    for row in rows:
        doc = row['Vedaant']
        for category in ['type', 'disease', 'pest']:
            for item in doc['predictions'][category]:
                record = {
                    'image': doc['image'],
                    'category': category,
                    'item_name': item[0],
                    'confidence': item[1],
                    'timestamp': pd.to_datetime(doc['timestamp']).date()  # Extract date portion only
                }
                data.append(record)
    return pd.DataFrame(data)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start_date = pd.to_datetime(request.form['start_date']).date()
        end_date = pd.to_datetime(request.form['end_date']).date()
        df = get_data()
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        return render_template('index8.html', tables=[filtered_df.to_html(classes='data')], titles=filtered_df.columns.values)
    return render_template('index8.html')

if __name__ == '__main__':
    app.run(debug=True)
