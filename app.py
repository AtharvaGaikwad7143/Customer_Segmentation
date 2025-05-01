
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.segment_model import load_data, preprocess_data, perform_clustering, get_segment_stats

st.set_page_config(page_title="📊 Customer Segmentation", layout="centered")

st.title("📊 Customer Segmentation (Marketing & E-commerce)")
st.write("Segment your customers based on purchasing behavior using K-Means clustering.")

# Load and preview data
data_path = "data/customers.csv"
try:
    data = load_data(data_path)
except FileNotFoundError:
    st.error("Please upload 'customers.csv' into the data folder.")
    st.stop()

st.subheader("🧾 Data Preview")
st.dataframe(data.head())

# Preprocess and cluster
rfm = preprocess_data(data)
rfm_clustered, model = perform_clustering(rfm)

st.subheader("📌 Customer Segments")
st.dataframe(rfm_clustered.head())

# Show cluster counts
segment_stats = get_segment_stats(rfm_clustered)
st.bar_chart(segment_stats['count'])

# Visualization
st.subheader("📉 Cluster Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=rfm_clustered, x='Recency', y='Monetary', hue='Cluster', palette='Set2', ax=ax)
st.pyplot(fig)
