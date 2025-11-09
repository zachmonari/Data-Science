#important libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
# Consider using ROC curve to find optimal threshold
from sklearn.metrics import roc_curve


#The dataset
url = "https://raw.githubusercontent.com/rene-gith/water-potability/main/water_potability.csv"
df = pd.read_csv(url)
print("Rows, Cols:", df.shape)

# Quick sanity checks
print("Label counts:\n", df["Potability"].value_counts(dropna=False))
print("\nMissing values per column:\n", df.isna().sum())

# Split features/labels and test set
X = df.drop(columns=["Potability"])
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)

# Build pipeline: impute → scale → model
pipe = make_pipeline(
   SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=0)
)
# Fit on Train only
pipe.fit(X_train, y_train)

# Predict on TEST + metrics
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
# Add these to your evaluation section
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Evaluation
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
optimal_threshold = thresholds[optimal_idx]
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Cross-validation score
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

print("UNIT7_ACC", round(acc, 3))
print("UNIT7_CONFUSION", cm.tolist())

# Tune threshold (Optional)
y_prob = pipe.predict_proba(X_test)[:, 1]
threshold = 0.6
y_pred_tuned = (y_prob >= threshold).astype(int)