# This is where we will write our program
# Data Science Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (Dataset)
df = pd.read_csv('user_behavior_dataset.csv')
print(df.columns)
# Examine first few rows
#pd.set_option('display.max_columns', None)
#print(df.head())


# Below are direct definitions from the kraggle dataset

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

# Visualizations


# View a histogram displaying how the column 'App Usage Time (min/day)' is distributed across the dataset.
# Right skewed, average of 271.13 minutes of app usage a day, as minutes increase instances decline.

plt.figure(figsize=(15, 7))
sns.histplot(df['App Usage Time (min/day)'], kde=True, bins=30)
# Add vertical lines for mean, median, min and max.
plt.axvline(df['App Usage Time (min/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['App Usage Time (min/day)'].mean():.2f}")
plt.axvline(df['App Usage Time (min/day)'].min(), color='green', linestyle='--', label=f"Min: {df['App Usage Time (min/day)'].min():.2f}")
plt.axvline(df['App Usage Time (min/day)'].max(), color='green', linestyle='--', label=f"Max: {df['App Usage Time (min/day)'].max():.2f}")
plt.axvline(df['App Usage Time (min/day)'].median(), color='green', linestyle='--', label=f"Median: {df['App Usage Time (min/day)'].median():.2f}")
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
# Add legend for lines.
plt.legend()
plt.show()


# View a histogram displaying how the column 'Screen On Time (hours/day)' is distributed across the dataset.
# Right skewed, average of 5.27 hours of screen time a day, as hours increase instances decline.

plt.figure(figsize=(15, 7))
sns.histplot(df['Screen On Time (hours/day)'], kde=True, bins=30)
# Add vertical lines for mean, median, min and max.
plt.axvline(df['Screen On Time (hours/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Screen On Time (hours/day)'].mean():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Screen On Time (hours/day)'].min():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Screen On Time (hours/day)'].max():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].median(), color='green', linestyle='--', label=f"Median: {df['Screen On Time (hours/day)'].median():.2f}")
plt.title('Distribution of Screen On Time (hours/day)')
plt.xlabel('Screen On Time (hours/day)')
plt.ylabel('Frequency')
# Add legend for lines.
plt.legend()
plt.show()


# View a histogram displaying how the column 'Data Usage (MB/day)' is distributed across the dataset.
# Right skewed, average of 929.74 MB of data usage a day, as MB increases instances decline.

plt.figure(figsize=(15, 7))
sns.histplot(df['Data Usage (MB/day)'], kde = True, bins = 30)
# Add vertical lines for mean, median, min and max.
plt.axvline(df['Data Usage (MB/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Data Usage (MB/day)'].mean():.2f}")
plt.axvline(df['Data Usage (MB/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Data Usage (MB/day)'].min():.2f}")
plt.axvline(df['Data Usage (MB/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Data Usage (MB/day)'].max():.2f}")
plt.axvline(df['Data Usage (MB/day)'].median(), color='green', linestyle='--', label=f"Median: {df['Data Usage (MB/day)'].median():.2f}")
plt.title('Distribution of Data Usage (MB/day)')
plt.xlabel('Data Usage (MB/day)')
plt.ylabel('Frequency')
# Add legend for lines.
plt.legend()
plt.show()


# View a histogram dispalying how the column 'Battery Drain (mAh/day)' is distibuted across the dataset.
# Slightly right skewed, average of 1525.16 mAhs battery drain a day, as mAH increases instances decline.

plt.figure(figsize=(15, 7))
sns.histplot(df['Battery Drain (mAh/day)'], kde = True, bins = 30)
# Add vertical liens for mean, median, min and max.
plt.axvline(df['Battery Drain (mAh/day)'].mean(), color = 'red', linestyle='--', label = f"Mean: {df['Battery Drain (mAh/day)'].mean():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].median(), color = 'green', linestyle='--', label = f"Median: {df['Battery Drain (mAh/day)'].median():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].min(), color = 'green', linestyle='--', label = f"Min: {df['Battery Drain (mAh/day)'].min():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].max(), color = 'green', linestyle='--', label = f"Max: {df['Battery Drain (mAh/day)'].max():.2f}")
plt.title('Distribution of Battery Drain (mAh/day)')
plt.xlabel('Battery Drain (mAh/day)')
plt.ylabel('Frequency')
# Add legend for lines.
plt.legend()
plt.show()