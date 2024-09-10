import streamlit as st
import joblib
import numpy as np
import os
from sklearn.cluster import MeanShift

# Load the KMeans model using the correct path
model_path = os.path.join(os.path.dirname(__file__), 'km_model.joblib')
kmeans_model = joblib.load(model_path)

# Option to choose the clustering algorithm
cluster_method = st.selectbox('Choose Clustering Method:', ['KMeans', 'Mean Shift'])

# Take inputs from the user using Streamlit
annual_income = st.number_input('Enter Annual Income (in $k):')
spending_score = st.number_input('Enter Spending Score (1-100):')

# Create an input array with the new data point
input_array = np.array([[annual_income, spending_score]])

# Button to trigger the prediction
if st.button('Check'):
    if cluster_method == 'KMeans':
        # KMeans prediction
        predicted_cluster = kmeans_model.predict(input_array)

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

    elif cluster_method == 'Mean Shift':
        # Sample data for MeanShift (add your own data here)
        data = np.array([[15, 39], [15, 81], [16, 6], [16, 77], [17, 40], [17, 76]])  # Example data, use your dataset here
        mean_shift_model = MeanShift()
        mean_shift_model.fit(data)

        # Predict the cluster using Mean Shift
        predicted_cluster = mean_shift_model.predict(input_array)

        # Get the label for Mean Shift (you may need to customize this based on your data)
        st.write(f"The new customer belongs to Mean Shift cluster: {predicted_cluster[0]}")
