### **Scikit-Learn: Concepts and Subconcepts**  

## **1. Core Concepts**  
- **Machine Learning Basics**  
  - Supervised Learning  
  - Unsupervised Learning  
  - Reinforcement Learning (limited support)  

- **Estimators**  
  - Classifiers  
  - Regressors  
  - Clustering Models  

- **Pipelines**  
  - Sequential Transformation  
  - Parameter Tuning  

- **Metrics and Scoring**  
  - Performance Evaluation  
  - Model Selection  

## **2. Data Preprocessing**  
- **Feature Scaling**  
  - Standardization  
  - Normalization  

- **Feature Selection**  
  - Filter Methods  
  - Wrapper Methods  
  - Embedded Methods  

- **Dimensionality Reduction**  
  - Principal Component Analysis (PCA)  
  - Linear Discriminant Analysis (LDA)  
  - Feature Agglomeration  

- **Data Transformation**  
  - Encoding Categorical Features  
  - Polynomial Features  
  - Power Transforms  

- **Handling Missing Values**  
  - Imputation  
  - Dropping Missing Data  

## **3. Model Selection & Validation**  
- **Cross-Validation**  
  - K-Fold  
  - Leave-One-Out  
  - Stratified K-Fold  

- **Hyperparameter Tuning**  
  - Grid Search  
  - Random Search  
  - Bayesian Optimization (via external libraries)  

- **Performance Metrics**  
  - Accuracy, Precision, Recall, F1-score  
  - ROC-AUC, PR-AUC  
  - Mean Squared Error (MSE), R² Score  

## **4. Supervised Learning**  
- **Regression Models**  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - ElasticNet  
  - Support Vector Regression (SVR)  
  - Decision Tree Regression  
  - Random Forest Regression  
  - Gradient Boosting Regression  
  - Neural Network Regression  

- **Classification Models**  
  - Logistic Regression  
  - Naïve Bayes  
  - k-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - Gradient Boosting Classifier  
  - Neural Network Classifier  

## **5. Unsupervised Learning**  
- **Clustering Algorithms**  
  - K-Means  
  - Hierarchical Clustering  
  - DBSCAN  
  - Mean Shift  
  - Gaussian Mixture Models (GMM)  

- **Anomaly Detection**  
  - Isolation Forest  
  - One-Class SVM  
  - Local Outlier Factor (LOF)  

## **6. Ensemble Methods**  
- **Bagging**  
  - Bagging Classifier  
  - Bagging Regressor  
  - Random Forest  

- **Boosting**  
  - AdaBoost  
  - Gradient Boosting (GBM)  
  - XGBoost (via external libraries)  

- **Stacking**  
  - Stacking Classifier  
  - Stacking Regressor  

- **Voting**  
  - Hard Voting  
  - Soft Voting  

## **7. Model Interpretation & Explainability**  
- **Feature Importance**  
  - Permutation Importance  
  - SHAP (via external libraries)  
  - LIME (via external libraries)  

- **Partial Dependence Plots**  
- **Calibration Curves**  

## **8. Feature Engineering**  
- **Categorical Encoding**  
  - One-Hot Encoding  
  - Label Encoding  
  - Target Encoding  

- **Interaction Features**  
- **Binning**  
- **Polynomial Features**  

## **9. Imbalanced Data Handling**  
- **Resampling Methods**  
  - Oversampling (SMOTE)  
  - Undersampling  
  - Class Weighting  

## **10. Neural Networks (via MLP in Scikit-Learn)**  
- **Multi-Layer Perceptron (MLP)**  
  - MLPClassifier  
  - MLPRegressor  

## **11. Dataset Utilities**  
- **Built-in Datasets**  
  - Iris  
  - Wine  
  - Breast Cancer  
  - Digits  

- **Data Loading**  
  - `load_*` functions  
  - `fetch_*` functions  

## **12. Parallelization & Performance Optimization**  
- **Joblib for Parallel Computing**  
- **Sparse Matrices for Large Datasets**  

---
---

## Scikit-learn Concepts

---

### 1. **Datasets**
   - **Built-in Datasets**
     - Iris, Wine, Breast Cancer, Digits, etc.
   - **Dataset Loaders**
     - `load_iris`, `load_digits`, etc.
   - **Dataset Generators**
     - `make_classification`, `make_regression`, `make_blobs`, etc.
   - **Dataset Utilities**
     - `fetch_openml`, `fetch_covtype`, `train_test_split`

---

### 2. **Preprocessing**
   - **Data Normalization**
     - `normalize`, `StandardScaler`, `MinMaxScaler`, etc.
   - **Categorical Encoding**
     - `OneHotEncoder`, `LabelEncoder`, `OrdinalEncoder`
   - **Feature Transformation**
     - `PolynomialFeatures`, `FunctionTransformer`
   - **Imputation**
     - `SimpleImputer`, `KNNImputer`, `MissingIndicator`
   - **Custom Preprocessing**
     - `TransformerMixin`

---

### 3. **Feature Selection**
   - **Univariate Selection**
     - `SelectKBest`, `SelectPercentile`
   - **Model-based Selection**
     - `SelectFromModel`
   - **Recursive Selection**
     - `RFE`, `RFECV`
   - **Variance Threshold**
     - `VarianceThreshold`

---

### 4. **Dimensionality Reduction**
   - **Linear Methods**
     - `PCA`, `SparsePCA`, `TruncatedSVD`
   - **Non-linear Methods**
     - `KernelPCA`, `LocallyLinearEmbedding (LLE)`
   - **Manifold Learning**
     - `Isomap`, `MDS`, `t-SNE`, `UMAP`
   - **Feature Extraction**
     - `DictionaryLearning`, `FactorAnalysis`

---

### 5. **Supervised Learning**
   - **Linear Models**
     - `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`
   - **Support Vector Machines**
     - `SVC`, `SVR`, `LinearSVC`
   - **Decision Trees**
     - `DecisionTreeClassifier`, `DecisionTreeRegressor`
   - **Ensemble Methods**
     - Random Forest (`RandomForestClassifier`, `RandomForestRegressor`)
     - Gradient Boosting (`GradientBoostingClassifier`, `GradientBoostingRegressor`)
     - Voting and Bagging (`VotingClassifier`, `BaggingClassifier`)
   - **Naive Bayes**
     - `GaussianNB`, `MultinomialNB`, `ComplementNB`
   - **Nearest Neighbors**
     - `KNeighborsClassifier`, `KNeighborsRegressor`

---

### 6. **Unsupervised Learning**
   - **Clustering**
     - `KMeans`, `DBSCAN`, `AgglomerativeClustering`
   - **Gaussian Mixture Models**
     - `GaussianMixture`, `BayesianGaussianMixture`
   - **Biclustering**
     - `SpectralBiclustering`, `SpectralCoclustering`
   - **Outlier Detection**
     - `IsolationForest`, `EllipticEnvelope`

---

### 7. **Model Evaluation**
   - **Scoring**
     - Accuracy, Precision, Recall, F1, AUC-ROC
   - **Cross-validation**
     - `cross_val_score`, `cross_validate`, `KFold`, `StratifiedKFold`
   - **Metrics**
     - Classification Metrics (`accuracy_score`, `f1_score`, `roc_auc_score`, etc.)
     - Regression Metrics (`mean_squared_error`, `r2_score`, etc.)
     - Clustering Metrics (`silhouette_score`, `adjusted_rand_score`)
   - **Hyperparameter Tuning**
     - `GridSearchCV`, `RandomizedSearchCV`

---

### 8. **Pipelines**
   - **Pipeline Construction**
     - `Pipeline`, `FeatureUnion`
   - **Pipeline Utilities**
     - `make_pipeline`, `make_union`

---

### 9. **Model Persistence**
   - **Serialization**
     - `joblib.dump`, `joblib.load`

---

### 10. **Miscellaneous**
   - **Custom Estimators**
     - Base Classes (`BaseEstimator`, `ClassifierMixin`, `RegressorMixin`)
   - **Utilities**
     - `check_array`, `safe_indexing`
   - **Parallelization**
     - `n_jobs` parameter for parallel processing

---

### 11. **Experimental Features**
   - **HistGradientBoosting**
     - `HistGradientBoostingClassifier`, `HistGradientBoostingRegressor`
   - **Neural Networks**
     - `MLPClassifier`, `MLPRegressor`

---
