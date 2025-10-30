import pandas as pd
students = pd.DataFrame({
    "Name": ["Ali", "Sara", "Tom", "Riya", "Jin", "Dan","Fin","Zac","Quin","Lin","Kev"],
    "Score": [85, 42, 77, 90, 66,50,38,80,75,56,68],
    "Attendance": [0.95, 0.80, 0.88, 0.99, 0.85,0.82,0.75,0.92,0.87,0.83,0.86]
})

#print(students.describe())
print(students)