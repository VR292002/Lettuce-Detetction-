from couchbase.cluster import Cluster, ClusterOptions
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
        'type': ', '.join([f"{item[0]}: {item[1]}" for item in doc['predictions']['type']]),
        'disease': ', '.join([f"{item[0]}: {item[1]}" for item in doc['predictions']['disease']]),
        'pest': ', '.join([f"{item[0]}: {item[1]}" for item in doc['predictions']['pest']]),
        'timestamp': doc['timestamp']
    }
    data.append(record)

# Convert list of dictionaries into pandas DataFrame
df = pd.DataFrame(data)

# Print DataFrame info
print(df.info())

# Display DataFrame using Streamlit
st.dataframe(df)
