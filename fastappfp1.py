from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.cluster import QueryOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd

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
        
        # Restructure DataFrame to have separate rows for type, disease, and pest
        new_rows = []
        for index, row in df.iterrows():
            for category in ['type', 'disease', 'pest']:
                for item, confidence in row['predictions'][category]:
                    new_rows.append({'image': row['image'], 'category': category, 'item': item, 'confidence': confidence, 'timestamp': row['timestamp']})
        new_df = pd.DataFrame(new_rows)
        
        # Order DataFrame by category (type, disease, pest)
        category_order = ['type', 'disease', 'pest']
        new_df['category'] = pd.Categorical(new_df['category'], categories=category_order, ordered=True)
        new_df = new_df.sort_values(by='category')
        
        # Render DataFrame as HTML table
        html_table = new_df.to_html(index=False)
        
        # Return HTML response directly
        return html_table
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
