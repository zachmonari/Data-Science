#Libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# ---------------------------
# 1. Sample dataset (3 features)
# ---------------------------
df = pd.DataFrame({
    "study_hours": [2, 3, 5, 8, 9, 1, 6, 7, 4, 10],
    "grade": [50, 55, 65, 80, 85, 45, 70, 78, 60, 90],
    "absences": [10, 8, 5, 3, 2, 12, 6, 4, 7, 1]
})
# ---------------------------
# 2. Scaling
# ---------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
# ---------------------------
# 3. K-Means clustering
# ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)
# ---------------------------
# 4. New student data
# ---------------------------
new_student = [[7, 72, 3]]  # study_hours, grade, absences
new_scaled = scaler.transform(new_student)

new_cluster = kmeans.predict(new_scaled)[0]
print("New student belongs to cluster:", new_cluster)
# ---------------------------
# 5. 3D Visualization
# ---------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Existing students
ax.scatter(
    scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2],
    c=kmeans.labels_, cmap="viridis", s=80
)
# Cluster centers
ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    kmeans.cluster_centers_[:, 2],
    c='red', s=300, marker='X', label='Cluster Centers'
)
# New student
ax.scatter(
    new_scaled[:, 0], new_scaled[:, 1], new_scaled[:, 2],
    c='black', s=200, marker='*', label='New Student'
)