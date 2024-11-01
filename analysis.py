# This is where we will write our program
# Data Science Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# Load the CSV file (Dataset)
df = pd.read_csv('user_behavior_dataset.csv')

# Examine first few rows
#pd.set_option('display.max_columns', None)
#print(df.head())


# Below are direct defi nitions from the kraggle dataset

# User ID: Unique identifier for each user.
# Device Model: Model of the user's smartphone.
# Operating System: The OS of the device (iOS or Android).
# App Usage Time: Daily time spent on mobile applications, measured in minutes.
# Screen On Time: Average hours per day the screen is active.
# Battery Drain: Daily battery consumption in mAh.
# Number of Apps Installed: Total apps available on the device.
# Data Usage: Daily mobile data consumption in megabytes.
# Age: Age of the user.
# Gender: Gender of the user (Male or Female).
# User Behavior Class: Classification of user behavior based on usage patterns (1 to 5).

# Summary of each column
#print(df.describe())

# Clean data --

# Check for null values
#print(df.isnull().sum())

# Check data types
#print(df.dtypes)

# Histograms Highlighting Min, Max, Median, and Mean --


# Histogram displaying how the column 'App Usage Time (min/day)' is distributed across the dataset.
# Right skewed, average of 271.13 minutes of app usage a day, as minutes increase instances decline.

#plt.figure(figsize=(15, 7))
#sns.histplot(df['App Usage Time (min/day)'], kde=True, bins=30)
# Add vertical lines for mean, median, min, and max.
#plt.axvline(df['App Usage Time (min/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['App Usage Time (min/day)'].mean():.2f}")
#plt.axvline(df['App Usage Time (min/day)'].min(), color='green', linestyle='--', label=f"Min: {df['App Usage Time (min/day)'].min():.2f}")
#plt.axvline(df['App Usage Time (min/day)'].max(), color='green', linestyle='--', label=f"Max: {df['App Usage Time (min/day)'].max():.2f}")
#plt.axvline(df['App Usage Time (min/day)'].median(), color='green', linestyle='--', label=f"Median: {df['App Usage Time (min/day)'].median():.2f}")
#plt.title('Distribution of App Usage Time (min/day)')
#plt.xlabel('App Usage Time (min/day)')
#plt.ylabel('Frequency')
# Add legend for lines.
#plt.legend()


# Histogram displaying how the column 'Screen On Time (hours/day)' is distributed across the dataset.
# Right skewed, average of 5.27 hours of screen time a day, as hours increase instances decline.

#plt.figure(figsize=(15, 7))
#sns.histplot(df['Screen On Time (hours/day)'], kde=True, bins=30)
# Add vertical lines for mean, median, min, and max.
#plt.axvline(df['Screen On Time (hours/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Screen On Time (hours/day)'].mean():.2f}")
#plt.axvline(df['Screen On Time (hours/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Screen On Time (hours/day)'].min():.2f}")
#plt.axvline(df['Screen On Time (hours/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Screen On Time (hours/day)'].max():.2f}")
#plt.axvline(df['Screen On Time (hours/day)'].median(), color='green', linestyle='--', label=f"Median: {df['Screen On Time (hours/day)'].median():.2f}")
#plt.title('Distribution of Screen On Time (hours/day)')
#plt.xlabel('Screen On Time (hours/day)')
#plt.ylabel('Frequency')
# Add legend for lines.
#plt.legend()


# Histogram displaying how the column 'Data Usage (MB/day)' is distributed across the dataset.
# Right skewed, average of 929.74 MB of data usage a day, as MB increases instances decline.

#plt.figure(figsize=(15, 7))
#sns.histplot(df['Data Usage (MB/day)'], kde = True, bins = 30)
# Add vertical lines for mean, median, min, and max.
#plt.axvline(df['Data Usage (MB/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Data Usage (MB/day)'].mean():.2f}")
#plt.axvline(df['Data Usage (MB/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Data Usage (MB/day)'].min():.2f}")
#plt.axvline(df['Data Usage (MB/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Data Usage (MB/day)'].max():.2f}")
#plt.axvline(df['Data Usage (MB/day)'].median(), color='green', linestyle='--', label=f"Median: {df['Data Usage (MB/day)'].median():.2f}")
#plt.title('Distribution of Data Usage (MB/day)')
#plt.xlabel('Data Usage (MB/day)')
#plt.ylabel('Frequency')
# Add legend for lines.
#plt.legend()


# Histogram displaying how the column 'Battery Drain (mAh/day)' is distibuted across the dataset.
# Slightly skewed right, average of 1525.16 mAhs battery drain a day, as mAH increases instances decline.

#plt.figure(figsize=(15, 7))
#sns.histplot(df['Battery Drain (mAh/day)'], kde = True, bins = 30)
# Add vertical liens for mean, median, min, and max.
#plt.axvline(df['Battery Drain (mAh/day)'].mean(), color = 'red', linestyle='--', label = f"Mean: {df['Battery Drain (mAh/day)'].mean():.2f}")
#plt.axvline(df['Battery Drain (mAh/day)'].median(), color = 'green', linestyle='--', label = f"Median: {df['Battery Drain (mAh/day)'].median():.2f}")
#plt.axvline(df['Battery Drain (mAh/day)'].min(), color = 'green', linestyle='--', label = f"Min: {df['Battery Drain (mAh/day)'].min():.2f}")
#plt.axvline(df['Battery Drain (mAh/day)'].max(), color = 'green', linestyle='--', label = f"Max: {df['Battery Drain (mAh/day)'].max():.2f}")
#plt.title('Distribution of Battery Drain (mAh/day)')
#plt.xlabel('Battery Drain (mAh/day)')
#plt.ylabel('Frequency')
# Add legend for lines.
#plt.legend()
#plt.show()


# Histogram displaying the distribution of 'Number of Apps Installed'.
# Slightly skewed right, average of 50.68 apps installed per user, as number of apps increase instances decline.

#plt.figure(figsize=(15, 7))
#sns.histplot(df['Number of Apps Installed'], kde=True, bins=30)
# Add vertical lines for mean, median, min, and max.
#plt.axvline(df['Number of Apps Installed'].mean(), color='red', linestyle='--', label=f"Mean: {df['Number of Apps Installed'].mean():.2f}")
#plt.axvline(df['Number of Apps Installed'].min(), color='green', linestyle='--', label=f"Min: {df['Number of Apps Installed'].min():.2f}")
#plt.axvline(df['Number of Apps Installed'].max(), color='green', linestyle='--', label=f"Max: {df['Number of Apps Installed'].max():.2f}")
#plt.axvline(df['Number of Apps Installed'].median(), color='green', linestyle='--', label=f"Median: {df['Number of Apps Installed'].median():.2f}")
#plt.title('Distribution of Number of Apps Installed')
#plt.xlabel('Number of Apps Installed')
#plt.ylabel('Frequency')
# Add legend for lines.
#plt.legend()
#plt.show()


# Histogram displaying the distribution of 'Age'.
# Symmetrical, average of 38.48 years of age, data remains consistent over the span of 18 to 59 years (age range of our dataset).

#plt.figure(figsize=(15, 7))
#sns.histplot(df['Age'], kde=True, bins=30)
# Add vertical lines for mean, median, min, and max.
#plt.axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.2f}")
#plt.axvline(df['Age'].min(), color='green', linestyle='--', label=f"Min: {df['Age'].min():.2f}")
#plt.axvline(df['Age'].max(), color='green', linestyle='--', label=f"Max: {df['Age'].max():.2f}")
#plt.axvline(df['Age'].median(), color='blue', linestyle='--', label=f"Median: {df['Age'].median():.2f}")
#plt.title('Distribution of Age')
#plt.xlabel('Age')
#plt.ylabel('Frequency')
# Add legend for lines
#plt.legend()
#plt.show() 


# Correlation Analysis --


# View scatterplots between every numeric feature combination to see linear relationships.
# This is very useful to see if there is consistent corrrelation betweeen features.


# Create scatterplot matrix for all numeric variables.
# Matrix is a pain in the ass to fit-to-screen for viewing, idk how to fix.

# Select only numeric columns for scatterplots.
# numeric_df = df.select_dtypes(include=[np.number])
# Increase the size of each plot in the pairplot.
# Adjust the aspect ratio (width-to-height ratio) to give more horizontal space.
#sns.pairplot(numeric_df, height=3, aspect=1)
#plt.show()

# View a heatmap displaying correlation between every pair of numeric variables. Values range -1 to 1.
# (Noted Relationships) Analysis --
# Higher user ID             -> no correlation vs any other numerical feature, expected.
# Higher app usage time      -> higher user behavior class, data usage, #apps installed, battery drain, screen time, app usage time.
# Higher screen on time      -> higher user behavior class, data usage, #apps installed, battery drain, app usage.
# Higher battery drain       -> higher user behavior class, data usage, #apps installed, screen time, app usage.
# Higher # apps installed    -> higher user behavior class, data usage, battery drain, screen time, app usage.
# Higher Data Usage          -> higher user behavior class, #apps installed, battery drain, screen time, app usage.
# Age                        -> Suprisingly no correlation vs any other numerical feature.
# Higher user behavior class -> higher data usage, #apps installed, battery drain, screen time, app usage.

# Create the correlation matrix for numeric variables
# corr_matrix = numeric_df.corr()

# Visualize the correlation matrix using a heatmap
# plt.figure(figsize=(10, 6))  
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap of Numeric Features')
# plt.show()


# Linear Regression Analysis --
# Define features (X) and target (y)
# Features X and target y
# X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
# y = df['Battery Drain (mAh/day)']
# Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize model
# model = LinearRegression()
# Train model
# model.fit(X_train, y_train)
# Lets predict using test data
# y_pred = model.predict(X_test)
# Lets see how our models doing
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R-Squared: {r2}")


# Below has our best fit model yet, check r2 and mse, this is for predicting battery drain
# Define features and target
# X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
# y = df['Battery Drain (mAh/day)']
# Initialize PolynomialFeatures for higher degree features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X)
# The shape of X_poly will be larger than X because it now includes the original features, squared features, and interaction terms
# print("Original X shape:", X.shape)
# print("X with Polynomial Features shape:", X_poly.shape)
# Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
# Linear regression model
# model = LinearRegression()
# Train model
# model.fit(X_train, y_train)
# Make predictions
# y_pred = model.predict(X_test)
# Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# Display
# print(f"Mean Squared Error with Polynomial Features: {mse}")
# print(f"R-Squared with Polynomial Features: {r2}")

# Perform 5-fold cross-validation on the training set and get R-Squared scores
# cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
# print("Cross-Validation R-Squared Scores:", cv_scores)
# print("Average R-Squared:", np.mean(cv_scores))

# Define features and target variable for training
X = df[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Data Usage (MB/day)', 'Number of Apps Installed']]
y = df['Battery Drain (mAh/day)']

# Transform features with polynomial expansion and fit the model
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Sample data for prediction
new_data = pd.DataFrame({
    'App Usage Time (min/day)': [393, 268, 154], 
    'Screen On Time (hours/day)': [6.4, 4.7, 4.0],
    'Data Usage (MB/day)': [1122, 944, 322],
    'Number of Apps Installed': [67, 42, 32]
})

# Transform new data using the same polynomial transformer
new_data_poly = poly.transform(new_data)
predictions = model.predict(new_data_poly)

# Combine samples with predictions for display
new_data['Predicted Battery Drain (mAh/day)'] = predictions
for i, row in new_data.iterrows():
    print(f"Sample {i + 1} - App Usage: {row['App Usage Time (min/day)']} min, "
          f"Screen Time: {row['Screen On Time (hours/day)']} hr, "
          f"Data Usage: {row['Data Usage (MB/day)']} MB, "
          f"Apps Installed: {row['Number of Apps Installed']}, "
          f"Predicted Drain: {row['Predicted Battery Drain (mAh/day)']:.2f} mAh")