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