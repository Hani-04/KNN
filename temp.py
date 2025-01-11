# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:28:09 2024

@author: 2215639
"""

# Import libraries
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Check libraries imported correctly
print('sklearn: %s' % sklearn.__version__)
print('numpy: %s' % np.__version__)
print('pandas: %s' % pd.__version__)
print('seaborn: %s' % sns.__version__)

# Import libraries from scikit-learn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE  # For oversampling
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load the dataset
file_path = "C:/Users/hanis/anaconda3/envs/ee3648_assignment/data.csv"
data = pd.read_csv(file_path)

# Section 1: Data Exploration
# -----------------------------------
print("\n*** Data Exploration ***")

# Basic information about the dataset
print(data.info())

# Generate summary statistics
summary_stats = data[['co', 'humidity', 'gas', 'smoke', 'temp']].describe()
print("\nSummary Statistics:")
print(summary_stats)

# Display the first few rows
print("First few rows of the dataset:\n", data.head())

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# TARGET VARIABLE ANALYSIS (motion)
# Check class distribution
print("Motion Distribution:\n", data['motion'].value_counts())

# Visualise the motion distribution
sns.countplot(x='motion', data=data)
plt.title("Motion Distribution")
plt.xlabel("Motion Detected")
plt.ylabel("Count")
plt.show()

# NUMERICAL FEATURE ANALYSIS
# Plot histograms for numerical features
data[['co', 'humidity', 'gas', 'smoke', 'temp']].hist(bins=30, figsize=(12, 8))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Generate pairwise plot for selected numerical features
sns.pairplot(data, vars=['co', 'humidity', 'gas',  'smoke', 'temp'], hue='motion', diag_kind='kde')
plt.show()

# CORRELATION ANALYSIS
# Select only numeric columns for correlation analysis
numeric_data = data.select_dtypes(include=['float64', 'int64', 'bool'])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap Original")
plt.show()

# Section 2: Data Pre-processing
# -----------------------------------
print("\n*** Data Pre-processing ***")

# Handle Missing Values
num_features = ['co', 'humidity', 'gas', 'smoke', 'temp']
imputer = SimpleImputer(strategy='mean')
data[num_features] = imputer.fit_transform(data[num_features])
print("\nMissing values handled. Updated numerical data preview:\n", data[num_features].head())

# HANDLING REDUNDANT FEATURES
# Drop the 'gas' feature due to perfect correlation with 'co'
data_cleaned = data.drop('gas', axis=1)
print("\nDropped 'gas' due to perfect correlation with 'co'.")

# Update num_features to exclude 'gas'
num_features.remove('gas')

# ENCODE CATEGORICAL VALUES
# One-hot encode the 'device' column
data_cleaned = pd.get_dummies(data_cleaned, columns=['device'], drop_first=True)
print("\nDevice column encoded into numerical format.")

# SCALE NUMERICAL FEATURES
# Standardise numerical columns
scaler = StandardScaler()
data_cleaned[num_features] = scaler.fit_transform(data_cleaned[num_features])
print("\nNumerical features scaled. Updated data preview:\n", data_cleaned[num_features].head())

# Updated correlation heatmap after dropping 'gas'
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])
correlation_matrix_updated = numeric_columns.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_updated, annot=True, cmap="coolwarm")
plt.title("Updated Correlation Heatmap (Without 'gas')")
plt.savefig("updated_correlation_heatmap.png")
plt.show()

# HANDLE CLASS IMBALANCE WITH SMOTE
X = data_cleaned.drop('motion', axis=1)
y = data_cleaned['motion']
smote = SMOTE(random_state=99)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Check the new class distribution after SMOTE
print("\nClass distribution after SMOTE:\n", pd.Series(y_balanced).value_counts())

# Section 3: Model Training and Evaluation
# -----------------------------------
print("\n*** Model Training and Evaluation ***")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=99)
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# IMPLEMENT KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("\nKNN model trained successfully.")

# MAKE PREDICTIONS
y_pred = knn.predict(X_test)
print("\nPredictions completed.")

# EVALUATE MODEL
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Visual Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",
            xticklabels=["No Motion", "Motion Detected"], 
            yticklabels=["No Motion", "Motion Detected"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# HYPERPARAMETER TUNING
print("\n*** Hyperparameter Tuning ***")
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy Score:", grid_search.best_score_)

# RETRAIN WITH OPTIMAL HYPERPARAMETERS
best_knn = grid_search.best_estimator_
best_knn.fit(X_train, y_train)
y_pred_optimised = best_knn.predict(X_test)
print("Accuracy of Optimised KNN:", accuracy_score(y_test, y_pred_optimised))
print("Classification Report of Optimised KNN:\n", classification_report(y_test, y_pred_optimised))

# CROSS-VALIDATION
cv_scores = cross_val_score(best_knn, X_balanced, y_balanced, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

# PLOT ROC CURVE
from sklearn.metrics import roc_curve, roc_auc_score
y_prob = best_knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.2f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()

