
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Basic RFM-like feature engineering
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    latest = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def perform_clustering(rfm, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans

def get_segment_stats(rfm_clustered):
    return rfm_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'count'})
