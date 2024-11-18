# Classification Analysis
# Script contains predictive modeling by means of logistic regression, and decision trees.
# Current able predictions - User Behavior Class

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# Load dataset
df = pd.read_csv("user_behavior_dataset.csv")

# Logistic Regression for User Behavior Class Prediction
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)']]
y = df['User Behavior Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"User Behavior Class Prediction - Accuracy: {accuracy:.2f}")

# Decision Tree for Operating System Prediction
# Encoding for categorical feature
df['Operating System'] = df['Operating System'].astype('category').cat.codes
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 
        'Battery Drain (mAh/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
y = df['Operating System']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree - Operating System Prediction Accuracy: {accuracy:.2f}")
print("Decision Tree - Classification Report:")
print(classification_report(y_test, y_pred))
cv_scores = cross_val_score(dt_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validated Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validated Accuracy: {np.mean(cv_scores):.2f}")
feature_importance = pd.Series(dt_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', figsize=(10, 6), title="Feature Importance")
plt.show()

# Random Forest method for Operating System 
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)']]
y = df['Operating System']  # 0 and 1 are OS categories
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
forest_model = RandomForestClassifier(n_estimators=100, random_state=42)  
forest_model.fit(X_train, y_train)
y_pred = forest_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest - Balanced Dataset Accuracy: {accuracy:.2f}")
print("Random Forest - Classification Report:")
print(classification_report(y_test, y_pred))
cv_scores = cross_val_score(forest_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"Cross-Validated Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validated Accuracy: {np.mean(cv_scores):.2f}")

