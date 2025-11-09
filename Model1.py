import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Feature importance
importance_df = (
    pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
      .sort_values('importance', ascending=False)
)
print(importance_df)

#Predict a new student
new_student = [[12, 3.1, 0.88, 0.82, 0.4*0.88 + 0.6*0.82]]
label = model.predict(new_student)[0]
proba = model.predict_proba(new_student)[0][1]

print("Prediction:", "Pass" if label==1 else "Fail")
print("Probability of passing:", round(proba, 2))

# --- 1. Feature Importance Visualization ---
plt.figure(figsize=(8,5))
sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis')
plt.title('Feature Importance in Predicting Student Success')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# --- 2. Confusion Matrix Visualization ---
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Fail', 'Predicted Pass'],
            yticklabels=['Actual Fail', 'Actual Pass'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Study Time Distribution ---
plt.figure(figsize=(6,4))
sns.countplot(x='studytime', data=student_data, palette='Blues')
plt.title("Study Time Distribution")
plt.xlabel("Weekly Study Time (1=low, 4=high)")
plt.ylabel("Number of Students")
plt.show()

# Relationship Between Study Time and Passing
plt.figure(figsize=(6,4))
sns.barplot(x='studytime', y=student_data['passed'].map({'yes':1, 'no':0}), data=student_data, palette='Greens')
plt.title("Passing Rate by Study Time")
plt.xlabel("Study Time Level")
plt.ylabel("Passing Rate")
plt.show()

#Failures vs Passing ---
plt.figure(figsize=(6,4))
sns.barplot(x='failures', y=student_data['passed'].map({'yes':1, 'no':0}), data=student_data, palette='Reds')
plt.title("Passing Rate by Number of Failures")
plt.xlabel("Number of Past Failures")
plt.ylabel("Passing Rate")
plt.show()

# Travel Time and Performance
plt.figure(figsize=(6,4))
sns.barplot(x='traveltime', y=student_data['passed'].map({'yes':1, 'no':0}), data=student_data, palette='Purples')
plt.title("Passing Rate by Travel Time to School")
plt.xlabel("Travel Time (1=short, 4=very long)")
plt.ylabel("Passing Rate")
plt.show()

# Absences vs Performance ---
plt.figure(figsize=(6,4))
sns.boxplot(x='passed', y='absences', hue='passed', data=student_data, palette='Set2', legend=False)
plt.title("Absences vs Pass Status")
plt.show()