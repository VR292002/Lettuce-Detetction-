from fastapi.responses import HTMLResponse
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.cluster import QueryOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd
import os
import streamlit as st

app = FastAPI()

# Couchbase connection settings
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant2')
collection = bucket.default_collection()


def fetch_documents_as_dataframe():
    try:
        query = f"SELECT * FROM Vedaant2"
        result = cluster.query(query, QueryOptions(adhoc=True))
        documents = []
        for row in result.rows():
            doc = row['Vedaant2']
            individual_data = doc['individual_data']
            avg_all = doc['avg_all']
            
            row_data = {
                'image_path1': individual_data[0]['image_path'],
                'image_path2': individual_data[1]['image_path'],
                'image_path3': individual_data[2]['image_path'],
                'lettuce1': [[entry[0],entry[1]] for entry in individual_data[0]['lettuce']],
                'lettuce2': [[entry[0],entry[1]] for entry in individual_data[1]['lettuce']],
                'lettuce3': [[entry[0],entry[1]] for entry in individual_data[2]['lettuce']],
                'disease1': [[entry[0],entry[1]] for entry in individual_data[0]['disease']],
                'disease2': [[entry[0],entry[1]] for entry in individual_data[1]['disease']],
                'disease3': [[entry[0],entry[1]] for entry in individual_data[2]['disease']],
                'pest1': [[entry[0],entry[1]] for entry in individual_data[0]['pest']],
                'pest2': [[entry[0],entry[1]] for entry in individual_data[1]['pest']],
                'pest3': [[entry[0],entry[1]] for entry in individual_data[2]['pest']],
                'average_percentage_lettuce': avg_all['average_percentage_lettuce'],
                'average_percentage_disease': avg_all['average_percentage_disease'],
                'average_percentage_pest': avg_all['average_percentage_pest']
            }
            
            documents.append(row_data)
        
        df = pd.DataFrame(documents)
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/", response_class=HTMLResponse)
async def display_table():
    try:
        # Fetch documents as DataFrame
        df = fetch_documents_as_dataframe()
        
        @app.get("/images/{image_path:path}")
        async def get_image(image_path: str):
            image_full_path = os.path.join("E:/", image_path)  # Assuming the images are stored in E:/
            if os.path.exists(image_full_path):
                return FileResponse(image_full_path)
            else:
                raise HTTPException(status_code=404, detail="Image not found")
        
        # Create a new column 'photos' to display images
        df['photos1'] = df.apply(lambda row: f'<img src="/images/{row["image_path1"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photos2'] = df.apply(lambda row: f'<img src="/images/{row["image_path2"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photos3'] = df.apply(lambda row: f'<img src="/images/{row["image_path3"]}" style="max-width:100px; max-height:100px;">', axis=1)
        # 
        
        
        # Render DataFrame as HTML table
        html_table = df.to_html(index=False, escape=False)
        html_table = html_table.replace('<th>', '<th style="text-align:center">')
        
        # Return HTML response directly
        return html_table
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


