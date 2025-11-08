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