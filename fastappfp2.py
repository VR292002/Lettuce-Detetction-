
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.cluster import QueryOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd
import os

app = FastAPI()

# Couchbase connection settings
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant')
collection = bucket.default_collection()

# Function to fetch documents and convert to DataFrame
def fetch_documents_as_dataframe():
    try:
        query = f"SELECT * FROM Vedaant"
        result = cluster.query(query, QueryOptions(adhoc=True))
        documents = []
        for row in result.rows():
            documents.append(row['Vedaant'])  # Extract 'Vedaant' key from each row
        df = pd.DataFrame(documents)
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    # Create Primary Index if it does not exist
    query = f"CREATE PRIMARY INDEX ON Vedaant"
    try:
        result = cluster.query(query, QueryOptions(adhoc=True))
        for row in result.rows():
            pass
    except Exception as e:
        print("Primary index already exists")

@app.get("/", response_class=HTMLResponse)
async def display_table():
    try:
        # Fetch documents as DataFrame
        df = fetch_documents_as_dataframe()
        
        # Serve images dynamically
        @app.get("/images/{image_path:path}")
        async def get_image(image_path: str):
            image_full_path = os.path.join("E:/", image_path)  # Assuming the images are stored in E:/
            if os.path.exists(image_full_path):
                return FileResponse(image_full_path)
            else:
                raise HTTPException(status_code=404, detail="Image not found")
        
        # Create a new column 'photos' to display images
        df['photos'] = df.apply(lambda row: f'<img src="/images/{row["image"]}" style="max-width:100px; max-height:100px;">', axis=1)
        
        # Move the 'image' column to the first position
        columns = ['image'] + [col for col in df.columns if col != 'image']
        df = df[columns]
        
        # Restructure DataFrame to have separate rows for type, disease, and pest
        new_rows = []
        for index, row in df.iterrows():
            for category in ['type', 'disease', 'pest']:
                for item, confidence in row['predictions'][category]:
                    new_rows.append({'category': category, 'item': item, 'confidence': confidence, 'timestamp': row['timestamp'], 'photos': row['photos']})
        new_df = pd.DataFrame(new_rows)
        
        # Order DataFrame by category (type, disease, pest)
        category_order = ['type', 'disease', 'pest']
        new_df['category'] = pd.Categorical(new_df['category'], categories=category_order, ordered=True)
        new_df = new_df.sort_values(by='category')
        
        # Render DataFrame as HTML table
        html_table = new_df.to_html(index=False, escape=False)
        html_table = html_table.replace('<th>', '<th style="text-align:center">')
        
        # Return HTML response directly
        return html_table
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
