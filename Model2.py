#important libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

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

print("UNIT7_ACC", round(acc, 3))
print("UNIT7_CONFUSION", cm.tolist())

# Tune threshold
y_prob = pipe.predict_proba(X_test)[:, 1]
threshold = 0.6
y_pred_tuned = (y_prob >= threshold).astype(int)