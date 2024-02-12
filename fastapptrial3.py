from fastapi.responses import HTMLResponse
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
bucket = cluster.bucket('Vedaant3')
collection = bucket.default_collection()


def fetch_documents_as_dataframe(start_date=None, end_date=None):
    try:
        query = f"SELECT * FROM Vedaant3"
        result = cluster.query(query, QueryOptions(adhoc=True))
        documents = []
        for row in result.rows():
            doc = row['Vedaant3']
            metadata = doc['metadata']
            individual_data = doc['individual_data']
            avg_all = doc['avg_all']
            
            # Extracting date from the timestamp and ignoring the time part
            timestamp_date = doc['timestamp'].split('T')[0]

            # Filtering based on selected start and end dates
            if start_date and end_date:
                if start_date <= timestamp_date <= end_date:
                    pass
                else:
                    continue

            row_data = {
                'id': metadata['id'],
                'image_path1': individual_data['img1']['image_path'],
                'image_path2': individual_data['img2']['image_path'],
                'image_path3': individual_data['img3']['image_path'],
                'image_path4': individual_data['img4']['image_path'],
                'image_path5': individual_data['img5']['image_path'],
                'lettuce1': individual_data['img1']['lettuce'],
                'lettuce2': individual_data['img2']['lettuce'],
                'lettuce3': individual_data['img3']['lettuce'],
                'lettuce4': individual_data['img4']['lettuce'],
                'lettuce5': individual_data['img5']['lettuce'],
                'disease1': individual_data['img1']['disease'],
                'disease2': individual_data['img2']['disease'],
                'disease3': individual_data['img3']['disease'],
                'disease4': individual_data['img4']['disease'],
                'disease5': individual_data['img5']['disease'],
                'pest1': individual_data['img1']['pest'],
                'pest2': individual_data['img2']['pest'],
                'pest3': individual_data['img3']['pest'],
                'pest4': individual_data['img4']['pest'],
                'pest5': individual_data['img5']['pest'],
                'average_percentage_lettuce': avg_all['average_percentage_lettuce'],
                'average_percentage_disease': avg_all['average_percentage_disease'],
                'average_percentage_pest': avg_all['average_percentage_pest'],
                'timestamp': timestamp_date,
                'create_dt': doc['create_dt'],
                'create_by': doc['create_by'],
                'update_dt': doc['update_dt'],
                'update_by': doc['update_by'],
                'affected_dt': doc['affected_dt']
            }
            
            documents.append(row_data)
        
        df = pd.DataFrame(documents)
        
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    # Create Primary Index if it does not exist
    query = f"CREATE PRIMARY INDEX ON Vedaant3"
    try:
        result = cluster.query(query, QueryOptions(adhoc=True))
        for row in result.rows():
            pass
    except Exception as e:
        print("Primary index already exists")

@app.get("/", response_class=HTMLResponse)
async def display_table(start_date: str = None, end_date: str = None):
    try:
        # HTML form for date selection
        date_form = """
        <form method="get">
            <label for="start_date">Start Date:</label>
            <input type="date" id="start_date" name="start_date"><br><br>
            <label for="end_date">End Date:</label>
            <input type="date" id="end_date" name="end_date"><br><br>
            <input type="submit" value="Filter">
        </form>
        """

        # Fetch documents as DataFrame
        df = fetch_documents_as_dataframe(start_date, end_date)

        if df.empty:
            return date_form + "<p>No results found within the selected date range.</p>"

        @app.get("/images/{image_path:path}")
        async def get_image(image_path: str):
            image_full_path = os.path.join("E:/", image_path)  # Assuming the images are stored in E:/
            if os.path.exists(image_full_path):
                return FileResponse(image_full_path)
            else:
                raise HTTPException(status_code=404, detail="Image not found")

        # Create a new column 'photos' to display images
        df['photo1'] = df.apply(lambda row: f'<img src="/images/{row["image_path1"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photo2'] = df.apply(lambda row: f'<img src="/images/{row["image_path2"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photo3'] = df.apply(lambda row: f'<img src="/images/{row["image_path3"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photo4'] = df.apply(lambda row: f'<img src="/images/{row["image_path4"]}" style="max-width:100px; max-height:100px;">', axis=1)
        df['photo5'] = df.apply(lambda row: f'<img src="/images/{row["image_path5"]}" style="max-width:100px; max-height:100px;">', axis=1)

        # Render DataFrame as HTML table
        html_table = df.to_html(index=False, escape=False)
        html_table = html_table.replace('<th>', '<th style="text-align:center">')

        # Return HTML response with form and table
        return date_form + html_table

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
