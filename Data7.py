# K-Means Clustering in Python
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

# Step 3: Visualize clusters
plt.figure(figsize=(8,6))
plt.scatter(df['study_hours'], df['grades'], c=df['cluster'], cmap='viridis', s=100)
plt.xlabel("Study Hours")
plt.ylabel("Grades")
plt.title("Student Clusters based on Study Habits")
plt.show()

print(df)
#Cluster 0 → High study hours & high grades

#Cluster 1 → Average study hours & grades

#Cluster 2 → Low study hours & low grades
# --- Step 2: New student data ---
new_student = pd.DataFrame({
    'study_hours': [7],
    'grades': [72]
})
# --- Step 3: Standardize new data using the same scaler ---
new_scaled = scaler.transform(new_student)

# --- Step 4: Predict cluster ---
predicted_cluster = kmeans.predict(new_scaled)
print(f"🎯 The new student belongs to cluster: {predicted_cluster[0]}")

# elbow method
# Test different K values
inertia_values = []
K_range = range(1, 10)
