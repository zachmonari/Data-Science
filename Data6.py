import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Example dataset
data = {'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male']}
df = pd.DataFrame(data)

# Count how many of each gender
gender_counts = df['gender'].value_counts()

# Plot pie chart
plt.figure(figsize=(5,5))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
plt.title('Gender Distribution')
plt.show()

# Example dataset
data = {
    'studytime': [1, 2, 3, 4, 2, 3, 1],
    'absences': [2, 6, 3, 0, 4, 1, 8],
    'failures': [0, 1, 0, 0, 1, 0, 2],
    'grade': [15, 13, 18, 19, 14, 17, 10]
}

df = pd.DataFrame(data)

# Compute correlation matrix
corr_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
