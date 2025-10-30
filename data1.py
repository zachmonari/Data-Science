import pandas as pd

students = pd.DataFrame({
    "Name": ["Ali", "Sara", "Tom", "Riya", "Jin"],
    "Score": [85, 42, 77, 90, 66],
    "Attendance": [0.95, 0.80, 0.88, 0.99, 0.85]
})

print(students.describe())

# Filter
passed = students[students["Score"] >= 50]

# Categorize
students["Category"] = pd.cut(
    students["Score"],
    bins=[0, 50, 75, 100],
    labels=["Fail", "Average", "Excellent"]
)

print(students)