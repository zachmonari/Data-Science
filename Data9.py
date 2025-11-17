import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    "study_hours": [2, 3, 1, 5, 8, 7, 4, 9, 6, 10],
    "previous_grade": [45, 50, 40, 60, 80, 78, 55, 90, 70, 95],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 1 = Pass, 0 = Fail
}

df = pd.DataFrame(data)
print(df)

X = df[["study_hours", "previous_grade"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
