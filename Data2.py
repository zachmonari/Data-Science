import pandas as pd
students = pd.DataFrame({
    "Name": ["Ali", "Sara", "Tom", "Riya", "Jin", "Dan","Fin","Zac","Quin","Lin","Kev"],
    "Score": [85, 42, 77, 90, 66,50,38,80,75,56,68],
    "Attendance": [0.95, 0.80, 0.88, 0.99, 0.85,0.82,0.75,0.92,0.87,0.83,0.86]
})

print(students.describe())
print(students)
print(students.shape)
print(students.info)

# Select a column
print(students["Attendance"])

# Filter rows
high_scores = students["Score"] > 70
print(high_scores)

# Sort by score
sorted_scores=students.sort_values(by="Score",ascending=False)
print(sorted_scores)

# Add new column
students["Passed"]=students["Score"] >= 50
print(students)

# Adjust scores
students["Adjusted"]=students["Score"] *1.05
print(students)

students["Category"] = pd.cut(
    students["Score"],
    bins=[0, 39,49,59, 69, 100],
    labels=["Fail","D", "C","B", "A"]
)
print(students)
print(students.describe())
print(students.max())
print(students.min())
