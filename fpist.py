import streamlit as st
import pandas as pd
import requests
import ast



# Fetch JSON data from the FastAPI server
response = requests.get("http://localhost:8000/")
if response.status_code == 200:
    data = response.json()

 


 

  
    # Provided JSON data as a string
    json_string = "{'image': 'E:/detection__images/vve.jpg', 'predictions': {'type': [['Iceberg', 0.759696364402771], ['Romaine', 0.16055086255073547], ['Green Batavia', 0.06451728194952011]], 'disease': [['Black Rot Leaf Spot', 0.45737096667289734], ['Mosaic Virus', 0.36322689056396484], ['Healthy', 0.14941924810409546]], 'pest': [['Healthy', 0.5694437026977539], ['Leafminers', 0.4031878709793091], ['Hornworms', 0.02432439476251602]]}, 'timestamp': '2024-02-07 15:59:29'}"

    # Convert the JSON string to a dictionary
    data = ast.literal_eval(json_string)

    # Create DataFrames for each category of predictions
    type_df = pd.DataFrame(data['predictions']['type'], columns=['Type', 'Type_Probability'])
    disease_df = pd.DataFrame(data['predictions']['disease'], columns=['Disease', 'Disease_Probability'])
    pest_df = pd.DataFrame(data['predictions']['pest'], columns=['Pest', 'Pest_Probability'])

    # Combine the DataFrames into a single DataFrame
    combined_df = pd.concat([type_df, disease_df, pest_df], axis=1)

    # Add the 'image' and 'timestamp' data to the DataFrame
    combined_df['image'] = data['image']
    combined_df['timestamp'] = data['timestamp']

    # Reorder columns to make 'image' the first column
    combined_df = combined_df[['image'] + [col for col in combined_df.columns if col != 'image' and col != 'timestamp'] + ['timestamp']]

    st.dataframe(combined_df)
    


else:
    st.error("Failed to fetch data from the server.")