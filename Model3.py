import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    "study_hours": [2, 3, 1, 5, 8, 7, 4, 9, 6, 10],
    "previous_grade": [45, 50, 40, 60, 80, 78, 55, 90, 70, 95],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "previous_grade"]]
y = df["passed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# Train Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# make predictions
y_pred = tree.predict(X_test)
print("Predictions:", y_pred)

# Evaluate Performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
from sklearn import tree as tree_plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
tree_plot.plot_tree(tree, feature_names=["study_hours", "previous_grade"],
                    class_names=["Fail", "Pass"],
                    filled=True, rounded=True)
plt.show()
