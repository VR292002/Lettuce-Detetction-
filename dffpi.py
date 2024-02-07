from fastapi import FastAPI, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")

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
                    'timestamp': pd.to_datetime(doc['timestamp']).date(),  # Extract date portion only
                    'photo': f'<img src="{doc["image"]}" width="200" height="200">'  # Add photo column with image HTML
                }
                data.append(record)
    return pd.DataFrame(data)

# Route for the home page
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle form submission
@app.post("/")
async def filter_data(request: Request, start_date: str = Form(...), end_date: str = Form(...)):
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    df = get_data()
    filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    return templates.TemplateResponse("index.html", {"request": request, "filtered_df": filtered_df})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

