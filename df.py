from couchbase.cluster import Cluster, QueryOptions, ClusterOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd
import streamlit as st

# Connect to Couchbase
cluster = Cluster('couchbase://localhost', ClusterOptions(PasswordAuthenticator('Vedaant', 'lipiom')))
bucket = cluster.bucket('Vedaant')
collection = bucket.default_collection()

# Query documents and extract data
query = "SELECT * FROM Vedaant"
rows = cluster.query(query)

# Convert data into a list of dictionaries
data = []
for row in rows:
    doc = row['Vedaant']
    record = {
        'image': doc['image'],
        'type': doc['predictions']['type'],
        'disease': doc['predictions']['disease'],
        'pest': doc['predictions']['pest'],
        'timestamp': doc['timestamp']
    }
    data.append(record)

# Convert list of dictionaries into pandas DataFrame
df = pd.DataFrame(data)

# Print DataFrame
print(df)
print(df.info())

