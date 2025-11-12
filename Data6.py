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

# Dataset 3

data1 = {
    'Name': ['Zac', 'Kev', 'Dan', 'Ian', 'Jack', 'Lyn', 'Ann', 'Sam', 'Kim', 'Joy'],
    'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Female'],
    'StudyTime': [4, 2, 3, 4, 1, 3, 4, 2, 1, 3],
    'Absences': [3, 10, 4, 1, 8, 2, 0, 6, 12, 2],
    'Failures': [0, 2, 1, 0, 3, 0, 0, 1, 2, 0],
    'Grade': [90, 60, 75, 95, 50, 88, 92, 70, 55, 85]
}

df = pd.DataFrame(data1)
df["Added"]=df["Grade"]*1.05
df["Passed"]=df["Grade"]>=70
# Sort by grade
sorted_grades=df.sort_values(by="Grade",ascending=False)
print(sorted_grades)

df["Category"] = pd.cut(
    df["Grade"],
    bins=[0, 39,49,59, 69, 100],
    labels=["Fail","D", "C","B", "A"]
)
print(df)

# Scatter plot
plt.figure(figsize=(6,4))
plt.scatter(df["StudyTime"],df['Grade'],marker='o')
plt.title('Student data')
plt.xlabel('StudyTime')
plt.ylabel('Grades')
plt.grid(alpha=0.3)
plt.show()

#pie chart of passed vs failed students
Pass_counts=df["Passed"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(
    Pass_counts,
    labels=Pass_counts.index,
    autopct='%1.1f%%',
    colors=['green', 'red'],
    startangle=90,
    wedgeprops={'edgecolor': 'black'},
    explode=[0.01,0.01]
)
plt.title('Passlist distribution')
plt.show()

#pie chart of how study time is distributed among the students
Hours=df["StudyTime"].value_counts()
plt.figure(figsize=(5,5))
plt.pie(
    Hours,
    labels=Hours.index,
    autopct='%1.1f%%',
    colors=['green', 'red','blue','yellow'],
    startangle=90,
    wedgeprops={'edgecolor': 'black'},
    explode=[0.01,0.01,0.01,0.01]
)
plt.title('Study Hours distribution')
plt.show()

# Plot pass list distribution
plt.figure(figsize = (5,5))
Pass_counts.plot(kind='bar',color=["green","red"])
plt.title("Pass list Distribution")
plt.xlabel("Pass vs Fail")
plt.ylabel("Number of students")
plt.xticks(rotation=0)
plt.show()

# Grades distribution
print("\nGrades statistics:\n", df['Grade'].describe())
plt.figure(figsize = (6,4))
df['Grade'].hist(bins=8, color='lightgreen', edgecolor='black')
plt.title("Grades Distribution of Students")
plt.xlabel("Grades")
plt.ylabel("Frequency")
plt.show()

# correlation matrix
corr=df.corr(numeric_only=True)
plt.figure(figsize = (6,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# correlation matrix 2
corr=df[['StudyTime','Grade',"Absences","Failures"]].corr()
plt.figure(figsize = (6,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix 2")
plt.show()

# correlation matrix 3
features=df[['StudyTime','Grade',"Absences","Failures"]]
corr=features.corr()
plt.figure(figsize = (6,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix 3")
plt.show()

# bar plot
plt.figure(figsize=(6,4))
sns.barplot(x="Name",y="Grade",data=df,color="green")
plt.title("Student Grades")
plt.xticks(rotation=45)
plt.show()