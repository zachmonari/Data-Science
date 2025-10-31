import pandas as pd
import matplotlib.pyplot as plt
#load the data
df=pd.read_csv('student-data.csv')
"""
#display first 5 rows
print(df.head())
#data summary
print(df.info())
#statistical summary
print(df.describe())
#missing values
print(df.isna())
"""
# Data analysis
#demographic insights

# Gender distribution
gender_counts=df["sex"].value_counts()
print("Gender distribution:\n",gender_counts)

# Plot gender distribution
plt.figure(figsize = (10,10))
gender_counts.plot(kind='bar',color=["skyblue","lightgreen"])
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.show()
