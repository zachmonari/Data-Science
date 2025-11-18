import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

plt.figure(figsize=(7,5))
plt.scatter(df["study_hours"], df["previous_grade"],
            c=df["passed"], cmap="bwr", s=100)

plt.xlabel("Study Hours")
plt.ylabel("Previous Grade")
plt.title("Student Performance: Pass vs Fail")
plt.grid(True)
plt.show()

# Fit logistic regression using ONE feature
X_single = df[["study_hours"]]
y = df["passed"]

model_single = LogisticRegression()
model_single.fit(X_single, y)

# Create smooth curve values
study_range = np.linspace(df["study_hours"].min(),
                          df["study_hours"].max(), 200).reshape(-1, 1)

prob_curve = model_single.predict_proba(study_range)[:, 1]

plt.figure(figsize=(7,5))

# Scatter points
plt.scatter(df["study_hours"], df["passed"], c=df["passed"], cmap="bwr")

# Logistic curve
plt.plot(study_range, prob_curve, linewidth=3)

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression Curve")
plt.grid(True)
plt.show()

plt.figure(figsize=(7,5))
plt.scatter(df["study_hours"], df["previous_grade"],
            c=df["passed"], cmap="bwr", s=100)



