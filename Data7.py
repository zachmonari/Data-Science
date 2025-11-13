# libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample data: Students' study time and grades
data = {
    'study_hours': [2, 3, 4, 8, 10, 12, 1, 6, 9, 11],
    'grades': [40, 45, 50, 75, 80, 90, 35, 70, 85, 88]
}

df = pd.DataFrame(data)

# Step 1: Standardize the data
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

# Step 2: Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled)
df['cluster'] = kmeans.labels_