import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# FIX: Convert to DataFrame with proper feature names
new_student = pd.DataFrame([[6, 65]], columns=["study_hours", "previous_grade"])
prediction = model.predict(new_student)

print("Pass (1) or Fail (0):", prediction)

