import streamlit as st
import joblib
import numpy as np
import os

# Load the KMeans model using the correct path
model_path = os.path.join(os.path.dirname(__file__), 'km_model.joblib')
model = joblib.load(model_path)

# Take inputs from the user using Streamlit
annual_income = st.number_input('Enter Annual Income (in $k):')
spending_score = st.number_input('Enter Spending Score (1-100):')

# Create an input array with the new data point
input_array = np.array([[annual_income, spending_score]])

# Button to trigger the prediction
if st.button('Check'):
    predicted_cluster = model.predict(input_array)

    # Assign meaningful labels to each cluster
    if predicted_cluster[0] == 1:
        cluster_label = "Highest"
    elif predicted_cluster[0] == 2:
        cluster_label = "Median Upper"
    elif predicted_cluster[0] == 0:
        cluster_label = "Median"
    elif predicted_cluster[0] == 4:
        cluster_label = "Median Lower"
    elif predicted_cluster[0] == 3:
        cluster_label = "Lowest"

    st.write(f"The new customer belongs to: {cluster_label}")

