import pandas as pd
import matplotlib.pyplot as plt
#load the data
df=pd.read_csv('student-data.csv')

#display first 5 rows
print(df.head())
#data summary
print(df.info())
#statistical summary
print(df.describe())
#missing values
print(df.isna())

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

# Create labels with counts
labels = [f"{gender} ({count})" for gender, count in zip(gender_counts.index, gender_counts.values)]

# Plot pie chart
plt.figure(figsize=(6,6))
plt.pie(
    gender_counts,
    labels=labels,
    autopct='%1.1f%%',
    colors=['skyblue', 'salmon'],
    startangle=90,
    wedgeprops={'edgecolor': 'black'}
)
plt.title("Gender Distribution of Students")
plt.show()

# Age distribution
print("\nAge statistics:\n", df['age'].describe())
plt.figure(figsize = (6,4))
df['age'].hist(bins=8, color='lightgreen', edgecolor='black')
plt.title("Age Distribution of Students")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
