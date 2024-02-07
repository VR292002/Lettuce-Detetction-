import streamlit as st
from couchbase.cluster import Cluster, ClusterOptions
from couchbase.auth import PasswordAuthenticator
import pandas as pd

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
                    'timestamp': pd.to_datetime(doc['timestamp']).date()  # Extract date portion only
                }
                data.append(record)
    return pd.DataFrame(data)

# Get data from Couchbase
df = get_data()

# Sidebar for date range selection
st.sidebar.header('Date Range Filter')
start_date = st.sidebar.date_input("Start date", min(df['timestamp']))
end_date = st.sidebar.date_input("End date", max(df['timestamp']))

# Filter data based on date range
filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

# Display filtered data in a custom layout
st.write("## Filtered Data")
for index, row in filtered_df.iterrows():
    st.write("### Image")
    st.image(row['image'], use_column_width=True)
    st.write("### Data")
    st.write(f"**Category:** {row['category']}")
    st.write(f"**Item:** {row['item_name']}")
    st.write(f"**Confidence:** {row['confidence']}")
    st.write("---")
