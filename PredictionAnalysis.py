# Data Science Project - Prediction Analysis
# This script performs predictive modeling on selected variables (Battery Drain, Data Usage, Screen On Time).
# It includes linear and polynomial regression for numerical predictions, logistic regression for classification, 
# and model evaluation metrics such as MSE, RMSE, R2, accuracy, and classification reports.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

df = pd.read_csv("user_behavior_dataset.csv")

# Linear Regression for App Usage Time Prediction
X = df[['Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
y = df['App Usage Time (min/day)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"App Usage Time Prediction - MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}, R-Squared: {r2:.2f}")
# Residual Analysis for App Usage Time
residuals = y_test - y_pred
print("\nResiduals Analysis (App Usage Time Prediction):")
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution for App Usage Time Prediction")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Linear Regression for Battery Drain Prediction
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
y = df['Battery Drain (mAh/day)']
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Battery Drain Prediction - MSE: {mse:.2f}, R-Squared: {r2:.2f}")
# Residual Analysis for Battery Drain
residuals = y_test - y_pred
print("\nResiduals Analysis (Battery Drain Prediction):")
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution for Battery Drain Prediction")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Linear Regression for Data Usage Prediction
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed']]
y = df['Data Usage (MB/day)']
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Data Usage Prediction - MSE: {mse:.2f}, R-Squared: {r2:.2f}")
# Residual Analysis for Data Usage
residuals = y_test - y_pred
print("\nResiduals Analysis (Data Usage Prediction):")
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution for Data Usage Prediction")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Linear Regression for Screen On Time Prediction
X = df[['App Usage Time (min/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
y = df['Screen On Time (hours/day)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Screen On Time Prediction - MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}, R-Squared: {r2:.2f}")
# Residual Analysis for Screen On Time
residuals = y_test - y_pred
print("\nResiduals Analysis (Screen On Time Prediction):")
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution for Screen On Time Prediction")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

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
print("Classification Report:")
print(classification_report(y_test, y_pred))
