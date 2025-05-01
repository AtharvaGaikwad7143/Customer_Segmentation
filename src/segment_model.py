
import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def analyze_segments(df):
    # Basic group count by segment
    return df['segment'].value_counts()
