import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree as tree_plot
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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

plt.figure(figsize=(12, 8))
tree_plot.plot_tree(tree, feature_names=["study_hours", "previous_grade"],
                    class_names=["Fail", "Pass"],
                    filled=True, rounded=True)
plt.show()

# Decision Boundary Visualization
# Mesh grid
x_min, x_max = X["study_hours"].min() - 1, X["study_hours"].max() + 1
y_min, y_max = X["previous_grade"].min() - 5, X["previous_grade"].max() + 5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
preds = tree.predict(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))

# Decision region
plt.contourf(xx, yy, preds, levels=1, alpha=0.25, cmap="bwr")

# Scatter
plt.scatter(df["study_hours"], df["previous_grade"], c=df["passed"],
            s=100, cmap="bwr", edgecolor="black")

plt.xlabel("Study Hours")
plt.ylabel("Previous Grade")
plt.title("Decision Tree Classification Boundary")
plt.show()

# Hyper-parameter tuning
from sklearn.model_selection import GridSearchCV

# Define the model
tree = DecisionTreeClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 2, 3, 4, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [None, "sqrt", "log2"]
}
# Set up Grid Search
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,              # 5-fold cross validation
    scoring="accuracy",
    verbose=1,
    n_jobs=-1          # use all CPU cores
)