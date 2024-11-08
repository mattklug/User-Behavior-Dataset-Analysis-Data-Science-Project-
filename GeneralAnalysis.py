# Data Science Project - General Data Exploration and Visualization
# This script performs data analysis on the usage behavior dataset.
# It includes visualizations such as histograms with additional statistical lines, 
# scatterplot matrix and a heatmap for correlation insight.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file (Dataset)
df = pd.read_csv('user_behavior_dataset.csv')

# Histograms with annotated lines for mean, median, min, and max

# App Usage Time (min/day)
plt.figure(figsize=(15, 7))
sns.histplot(df['App Usage Time (min/day)'], kde=True, bins=30)
plt.axvline(df['App Usage Time (min/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['App Usage Time (min/day)'].mean():.2f}")
plt.axvline(df['App Usage Time (min/day)'].min(), color='green', linestyle='--', label=f"Min: {df['App Usage Time (min/day)'].min():.2f}")
plt.axvline(df['App Usage Time (min/day)'].max(), color='green', linestyle='--', label=f"Max: {df['App Usage Time (min/day)'].max():.2f}")
plt.axvline(df['App Usage Time (min/day)'].median(), color='blue', linestyle='--', label=f"Median: {df['App Usage Time (min/day)'].median():.2f}")
plt.title('Distribution of App Usage Time (min/day)')
plt.xlabel('App Usage Time (min/day)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Screen On Time (hours/day)
plt.figure(figsize=(15, 7))
sns.histplot(df['Screen On Time (hours/day)'], kde=True, bins=30)
plt.axvline(df['Screen On Time (hours/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Screen On Time (hours/day)'].mean():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Screen On Time (hours/day)'].min():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Screen On Time (hours/day)'].max():.2f}")
plt.axvline(df['Screen On Time (hours/day)'].median(), color='blue', linestyle='--', label=f"Median: {df['Screen On Time (hours/day)'].median():.2f}")
plt.title('Distribution of Screen On Time (hours/day)')
plt.xlabel('Screen On Time (hours/day)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Data Usage (MB/day)
plt.figure(figsize=(15, 7))
sns.histplot(df['Data Usage (MB/day)'], kde=True, bins=30)
plt.axvline(df['Data Usage (MB/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Data Usage (MB/day)'].mean():.2f}")
plt.axvline(df['Data Usage (MB/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Data Usage (MB/day)'].min():.2f}")
plt.axvline(df['Data Usage (MB/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Data Usage (MB/day)'].max():.2f}")
plt.axvline(df['Data Usage (MB/day)'].median(), color='blue', linestyle='--', label=f"Median: {df['Data Usage (MB/day)'].median():.2f}")
plt.title('Distribution of Data Usage (MB/day)')
plt.xlabel('Data Usage (MB/day)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Battery Drain (mAh/day)
plt.figure(figsize=(15, 7))
sns.histplot(df['Battery Drain (mAh/day)'], kde=True, bins=30)
plt.axvline(df['Battery Drain (mAh/day)'].mean(), color='red', linestyle='--', label=f"Mean: {df['Battery Drain (mAh/day)'].mean():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].min(), color='green', linestyle='--', label=f"Min: {df['Battery Drain (mAh/day)'].min():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].max(), color='green', linestyle='--', label=f"Max: {df['Battery Drain (mAh/day)'].max():.2f}")
plt.axvline(df['Battery Drain (mAh/day)'].median(), color='blue', linestyle='--', label=f"Median: {df['Battery Drain (mAh/day)'].median():.2f}")
plt.title('Distribution of Battery Drain (mAh/day)')
plt.xlabel('Battery Drain (mAh/day)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Number of Apps Installed
plt.figure(figsize=(15, 7))
sns.histplot(df['Number of Apps Installed'], kde=True, bins=30)
plt.axvline(df['Number of Apps Installed'].mean(), color='red', linestyle='--', label=f"Mean: {df['Number of Apps Installed'].mean():.2f}")
plt.axvline(df['Number of Apps Installed'].min(), color='green', linestyle='--', label=f"Min: {df['Number of Apps Installed'].min():.2f}")
plt.axvline(df['Number of Apps Installed'].max(), color='green', linestyle='--', label=f"Max: {df['Number of Apps Installed'].max():.2f}")
plt.axvline(df['Number of Apps Installed'].median(), color='blue', linestyle='--', label=f"Median: {df['Number of Apps Installed'].median():.2f}")
plt.title('Distribution of Number of Apps Installed')
plt.xlabel('Number of Apps Installed')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Age
plt.figure(figsize=(15, 7))
sns.histplot(df['Age'], kde=True, bins=30)
plt.axvline(df['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df['Age'].mean():.2f}")
plt.axvline(df['Age'].min(), color='green', linestyle='--', label=f"Min: {df['Age'].min():.2f}")
plt.axvline(df['Age'].max(), color='green', linestyle='--', label=f"Max: {df['Age'].max():.2f}")
plt.axvline(df['Age'].median(), color='blue', linestyle='--', label=f"Median: {df['Age'].median():.2f}")
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Scatterplot matrix to view relationships between numeric features
numeric_df = df.select_dtypes(include=[np.number])
sns.pairplot(numeric_df, height=3, aspect=1)
plt.show()

# Heatmap for correlation analysis
plt.figure(figsize=(10, 6))
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
