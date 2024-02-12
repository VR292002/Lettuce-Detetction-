import streamlit as st
import pandas as pd
import requests

# Fetch JSON data from the FastAPI server
response = requests.get("http://localhost:8000/")
if response.status_code == 200:
    data = response.json()
    # Extract data from each dictionary and store them in lists
    images = []
    timestamps = []
    diseases = []
    disease_probabilities = []
    pests = []
    pest_probabilities = []
    types = []
    type_probabilities = []

    for item in data:
        vedaant_data = item.pop('Vedaant')
        images.append(vedaant_data['image'])
        timestamps.append(vedaant_data['timestamp'])
        for disease, disease_prob in vedaant_data['predictions']['disease']:
            diseases.append(disease)
            disease_probabilities.append(disease_prob)
        for pest, pest_prob in vedaant_data['predictions']['pest']:
            pests.append(pest)
            pest_probabilities.append(pest_prob)
        for plant_type, type_prob in vedaant_data['predictions']['type']:
            types.append(plant_type)
            type_probabilities.append(type_prob)

    # Create DataFrames for each set of predictions
    disease_df = pd.DataFrame({'Disease': diseases, 'Disease_Probability': disease_probabilities})
    pest_df = pd.DataFrame({'Pest': pests, 'Pest_Probability': pest_probabilities})
    type_df = pd.DataFrame({'Type': types, 'Type_Probability': type_probabilities})

    # Combine the DataFrames
    combined_df = pd.concat([type_df, disease_df, pest_df], axis=1)

    # Convert timestamps list into DataFrame
    timestamp_df = pd.DataFrame({'timestamp': timestamps})

    # Concatenate timestamp_df with combined_df
    combined_df = pd.concat([combined_df, timestamp_df], axis=1)

    # Display the DataFrame
    st.dataframe(combined_df)




