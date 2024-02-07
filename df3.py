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
    for category in ['type', 'disease', 'pest']:
        for item in doc['predictions'][category]:
            record = {
                'image': doc['image'],
                'category': category,
                'item_name': item[0],
                'confidence': item[1],
                'timestamp': doc['timestamp']
            }
            data.append(record)

# Convert list of dictionaries into pandas DataFrame
df = pd.DataFrame(data)
df.groupby(["image"])

# Print DataFrame info
print(df.info())

# Display DataFrame using Streamlit
st.dataframe(df)
