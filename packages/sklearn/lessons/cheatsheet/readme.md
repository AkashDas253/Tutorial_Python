
# Scikit-learn (sklearn) Cheatsheet

## 1. Importing scikit-learn
- import numpy as np
- import pandas as pd
- from sklearn.model_selection import train_test_split  # Train-test split
- from sklearn.preprocessing import StandardScaler  # Feature scaling
- from sklearn.metrics import accuracy_score, classification_report  # Evaluation metrics
- from sklearn.pipeline import Pipeline  # Pipeline for preprocessing

## 2. Loading Datasets
- from sklearn.datasets import load_iris  # Load Iris dataset
- data = load_iris()  # Load data
- X, y = data.data, data.target  # Features and labels

## 3. Train-Test Split
- X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data

## 4. Feature Scaling
- scaler = StandardScaler()  # Create scaler
- X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
- X_test_scaled = scaler.transform(X_test)  # Transform test data

## 5. Classification Models
- from sklearn.linear_model import LogisticRegression  # Logistic Regression
- model = LogisticRegression()  # Initialize model
- model.fit(X_train_scaled, y_train)  # Train model
- y_pred = model.predict(X_test_scaled)  # Predict

## 6. Evaluation Metrics
- accuracy = accuracy_score(y_test, y_pred)  # Accuracy
- report = classification_report(y_test, y_pred)  # Classification report
- print(report)  # Print report

## 7. Regression Models
- from sklearn.linear_model import LinearRegression  # Linear Regression
- model = LinearRegression()  # Initialize model
- model.fit(X_train, y_train)  # Train model
- y_pred = model.predict(X_test)  # Predict

## 8. Model Selection
- from sklearn.model_selection import GridSearchCV  # Grid search for hyperparameters
- param_grid = {'C': [0.1, 1, 10]}  # Hyperparameter grid
- grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)  # Grid search object
- grid_search.fit(X_train_scaled, y_train)  # Fit grid search

## 9. Pipelines
- from sklearn.pipeline import Pipeline  # Pipeline for preprocessing
- pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])  # Create pipeline
- pipeline.fit(X_train, y_train)  # Fit pipeline
- y_pred = pipeline.predict(X_test)  # Predict using pipeline

## 10. Clustering Models
- from sklearn.cluster import KMeans  # K-Means Clustering
- model = KMeans(n_clusters=3)  # Initialize model
- model.fit(X)  # Fit model
- labels = model.labels_  # Cluster labels

## 11. Dimensionality Reduction
- from sklearn.decomposition import PCA  # Principal Component Analysis
- pca = PCA(n_components=2)  # Initialize PCA
- X_reduced = pca.fit_transform(X)  # Reduce dimensions

## 12. Saving and Loading Models
- from joblib import dump, load  # Joblib for saving/loading
- dump(model, 'model.joblib')  # Save model
- loaded_model = load('model.joblib')  # Load model
