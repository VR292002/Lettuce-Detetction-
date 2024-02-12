from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from database import process_data
from couchbase.cluster import Cluster, ClusterOptions, QueryOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd
import os
from datetime import datetime

app = FastAPI()

# Couchbase connection settings
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant3')
collection = bucket.default_collection()


# Endpoint for image classification
@app.post("/classify")
async def classify(data: dict):
    try:
        # Check if exactly 5 image paths are provided
        if "image_paths" not in data or len(data["image_paths"]) != 5:
            raise HTTPException(status_code=400, detail="Exactly 5 image paths are required.")

        # Ensure create_by is provided
        create_by = data.get("create_by")
        if not create_by:
            raise HTTPException(status_code=400, detail="create_by field is required.")

        # Process the data
        combined_data = process_data(data, create_by=create_by)

        # Generate a unique document ID by combining create_by and timestamp
        timestamp = datetime.now().isoformat()  # Get the current timestamp in ISO format
        doc_id = f"{create_by}_{timestamp}"  # Combine create_by and timestamp for document ID

        # Store the combined data in Couchbase using the generated ID
        collection.upsert(doc_id, combined_data)

        # Return a success response with the document ID
        return JSONResponse(content={"status": "success", "message": "Data processed and stored successfully.", "document_id": doc_id})
    except Exception as e:
        # Return an error response if an exception occurs
        return JSONResponse(content={"status": "error", "message": str(e)})
















# Function to fetch documents as DataFrame
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
            timestamp = datetime.fromisoformat(doc['timestamp'])
            date = timestamp.date()
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            row_data = {
                'id': metadata['id'],
                'image_path1': individual_data['img1']['image_path'],
                'image_path2': individual_data['img2']['image_path'],
                'image_path3': individual_data['img3']['image_path'],
                'image_path4': individual_data['img4']['image_path'],
                'image_path5': individual_data['img5']['image_path'],
                'lettuce1': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img1']['lettuce'].items()]),
                'lettuce2': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img2']['lettuce'].items()]),
                'lettuce3': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img3']['lettuce'].items()]),
                'lettuce4': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img4']['lettuce'].items()]),
                'lettuce5': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img5']['lettuce'].items()]),
                'disease1': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img1']['disease'].items()]),
                'disease2': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img2']['disease'].items()]),
                'disease3': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img3']['disease'].items()]),
                'disease4': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img4']['disease'].items()]),
                'disease5': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img5']['disease'].items()]),
                'pest1': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img1']['pest'].items()]),
                'pest2': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img2']['pest'].items()]),
                'pest3': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img3']['pest'].items()]),
                'pest4': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img4']['pest'].items()]),
                'pest5': "<br>".join([f"• {item}: {value}" for item, value in individual_data['img5']['pest'].items()]),
                'average_percentage_lettuce': "<br>".join([f"• {item}: {value}" for item, value in avg_all['average_percentage_lettuce'].items()]),
                'average_percentage_disease': "<br>".join([f"• {item}: {value}" for item, value in avg_all['average_percentage_disease'].items()]),
                'average_percentage_pest': "<br>".join([f"• {item}: {value}" for item, value in avg_all['average_percentage_pest'].items()]),
                'timestamp': doc['timestamp'],
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





# Endpoint to display table with images
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

        # Convert start_date and end_date to datetime objects
        start_date = datetime.fromisoformat(start_date) if start_date else None
        end_date = datetime.fromisoformat(end_date) if end_date else None

        # Fetch documents as DataFrame
        df = fetch_documents_as_dataframe(start_date, end_date)

        if df.empty:
            return date_form + "<p>No results found within the selected date range.</p>"

        # Endpoint to serve images
        @app.get("/images/{image_path:path}")
        async def get_image(image_path: str):
            image_full_path = image_path  # Using full image path provided in the JSON
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

        # Concatenate image paths
        df['image_paths'] = df.apply(lambda row: '<br><br><br><br>'.join(row[f'image_path{i}'] for i in range(1, 6)), axis=1)

        # Concatenate lettuce predictions
        df['lettuce_predictions'] = df.apply(lambda row: '<br><br>'.join(str(row[f'lettuce{i}']) for i in range(1, 6)), axis=1)

        # Concatenate disease predictions
        df['disease_predictions'] = df.apply(lambda row: '<br><br>'.join(str(row[f'disease{i}']) for i in range(1, 6)), axis=1)

        # Concatenate pest predictions
        df['pest_predictions'] = df.apply(lambda row: '<br><br>'.join(str(row[f'pest{i}']) for i in range(1, 6)), axis=1)

        # Drop individual image path and prediction columns
        df.drop(columns=[f'image_path{i}' for i in range(1, 6)], inplace=True)
        df.drop(columns=[f'lettuce{i}' for i in range(1, 6)], inplace=True)
        df.drop(columns=[f'disease{i}' for i in range(1, 6)], inplace=True)
        df.drop(columns=[f'pest{i}' for i in range(1, 6)], inplace=True)

        # Render DataFrame as HTML table
        html_table = df.to_html(index=False, escape=False)
        html_table = html_table.replace('<th>', '<th style="text-align:center">')

        # Return HTML response with form and table
        return date_form + html_table

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

