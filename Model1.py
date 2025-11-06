import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# the dataset
student_data = pd.DataFrame({
    'hours_studied_per_week': [15, 8, 20, 5, 12, 18, 3, 25, 10, 22],
    'previous_gpa'         : [3.5, 2.8, 3.8, 2.2, 3.2, 3.9, 2.0, 4.0, 3.0, 3.7],
    'attendance_rate'      : [0.95,0.70,0.98,0.60,0.85,0.92,0.55,1.00,0.80,0.96],
    'assignments_completed': [0.90,0.65,0.95,0.40,0.80,0.98,0.35,1.00,0.75,0.93],
    'passed'               : [1,0,1,0,1,1,0,1,0,1]
})
print(student_data.head())
print(student_data.describe())

#Feature engineering
student_data['engagement_score']=(
    student_data['attendance_rate']*0.4+
    student_data['assignments_completed']*0.6
)
print(student_data[['attendance_rate','assignments_completed','engagement_score']])

#Split into train/test
features = ['hours_studied_per_week','previous_gpa','attendance_rate',
            'assignments_completed','engagement_score']

x=student_data[features]
y=student_data['passed']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
print(x_train.shape)
print("Train size:", x_train.shape[0], " Test size:", x_test.shape[0])

#Train a Random Forest
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x_train,y_train)
print("Model trained!")

#Predict & evaluate
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)
print("Model Prediction Accuracy: ",accuracy)

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix [rows=true, cols=predicted]:")
print(confusion_matrix(y_test, y_pred))