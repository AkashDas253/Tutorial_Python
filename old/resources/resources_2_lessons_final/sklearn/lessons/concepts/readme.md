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
