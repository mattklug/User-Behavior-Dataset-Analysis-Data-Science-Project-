# Regression Analysis
# Script contains predictive modeling by means of regression.
# Current able predictions - App Usage Time, Battery Drain, Data Usage, and Screen on Time.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
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
# K-fold cross-validation
cv_mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"App Usage Time Cross-Validated MSE: {np.mean(cv_mse):.2f}, Cross-Validated R-Squared: {np.mean(cv_r2):.2f}")
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
# K-fold cross-validation
cv_mse = -cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
print(f"Battery Drain Cross-Validated MSE: {np.mean(cv_mse):.2f}, Cross-Validated R-Squared: {np.mean(cv_r2):.2f}")
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
# K-fold cross-validation
cv_mse = -cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
print(f"Data Usage Cross-Validated MSE: {np.mean(cv_mse):.2f}, Cross-Validated R-Squared: {np.mean(cv_r2):.2f}")
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
# K-fold cross-validation
cv_mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Screen On Time Cross-Validated MSE: {np.mean(cv_mse):.2f}, Cross-Validated R-Squared: {np.mean(cv_r2):.2f}")
# Residual Analysis for Screen On Time
residuals = y_test - y_pred
print("\nResiduals Analysis (Screen On Time Prediction):")
print(residuals.describe())
sns.histplot(residuals, kde=True)
plt.title("Residuals Distribution for Screen On Time Prediction")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()
