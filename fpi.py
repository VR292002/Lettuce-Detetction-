from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator

# Connect to Couchbase
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant')
collection = bucket.default_collection()

app = FastAPI()

def get_structured_data(doc_id):
    try:
        # Retrieve JSON data from Couchbase
        result = collection.get(doc_id)
        if result.success:
            data = result.content_as[str]
            return data
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/")
async def get_prediction():
    try:
        # Assuming the document ID is known
        doc_id = "image_0d8f3f07-13a8-4846-936d-03e2c49b7fc7"
        data = get_structured_data(doc_id)
        return JSONResponse(content=data, status_code=200)
    except HTTPException as http_error:
        return http_error