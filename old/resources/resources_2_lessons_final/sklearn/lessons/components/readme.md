# Modules

---

- **`sklearn.datasets`**  
  - `sklearn.datasets.load_iris` # Load the Iris dataset.
  - `sklearn.datasets.load_digits` # Load the Digits dataset.
  - `sklearn.datasets.load_boston` # Load the Boston housing dataset.
  - `sklearn.datasets.make_classification` # Generate a random classification problem.
  - `sklearn.datasets.make_blobs` # Generate isotropic Gaussian blobs for clustering.
  - `sklearn.datasets.fetch_openml` # Fetch datasets from OpenML.
  - `sklearn.datasets.fetch_covtype` # Fetch the Covertype dataset.

- **`sklearn.preprocessing`**  
  - `sklearn.preprocessing.StandardScaler` # Standardize features by removing the mean and scaling to unit variance.
  - `sklearn.preprocessing.MinMaxScaler` # Scale features to a given range (e.g., 0 to 1).
  - `sklearn.preprocessing.OneHotEncoder` # Encode categorical features as a one-hot numeric array.
  - `sklearn.preprocessing.LabelEncoder` # Encode target labels with values between 0 and n_classes-1.
  - `sklearn.preprocessing.OrdinaryEncoder` # Encode categorical features with an ordinal encoding.
  - `sklearn.preprocessing.PolynomialFeatures` # Generate polynomial and interaction features.
  - `sklearn.preprocessing.FunctionTransformer` # Apply a user-defined function to features.

- **`sklearn.feature_selection`**  
  - `sklearn.feature_selection.SelectKBest` # Select features based on the k highest scores.
  - `sklearn.feature_selection.RFE` # Recursive feature elimination.
  - `sklearn.feature_selection.SelectFromModel` # Select features based on importance weights from a model.
  - `sklearn.feature_selection.VarianceThreshold` # Remove features with low variance.

- **`sklearn.decomposition`**  
  - `sklearn.decomposition.PCA` # Principal component analysis for dimensionality reduction.
  - `sklearn.decomposition.NMF` # Non-negative matrix factorization.
  - `sklearn.decomposition.TruncatedSVD` # Dimensionality reduction using truncated SVD.
  - `sklearn.decomposition.Isomap` # Isometric feature mapping.
  - `sklearn.decomposition.MDS` # Multidimensional scaling.
  - `sklearn.decomposition.TSNE` # t-Distributed Stochastic Neighbor Embedding.

- **`sklearn.model_selection`**  
  - `sklearn.model_selection.train_test_split` # Split arrays or matrices into random train and test subsets.
  - `sklearn.model_selection.KFold` # K-Folds cross-validation.
  - `sklearn.model_selection.StratifiedKFold` # Stratified K-Folds cross-validation.
  - `sklearn.model_selection.cross_val_score` # Evaluate a score by cross-validation.
  - `sklearn.model_selection.GridSearchCV` # Exhaustive search over a specified parameter grid.
  - `sklearn.model_selection.RandomizedSearchCV` # Randomized search over parameters.

- **`sklearn.linear_model`**  
  - `sklearn.linear_model.LinearRegression` # Ordinary least squares linear regression.
  - `sklearn.linear_model.Ridge` # Ridge regression.
  - `sklearn.linear_model.Lasso` # Lasso regression.
  - `sklearn.linear_model.LogisticRegression` # Logistic regression.
  - `sklearn.linear_model.RidgeCV` # Ridge regression with built-in cross-validation.

- **`sklearn.svm`**  
  - `sklearn.svm.SVC` # Support vector classification.
  - `sklearn.svm.SVR` # Support vector regression.
  - `sklearn.svm.LinearSVC` # Linear support vector classification.
  - `sklearn.svm.LinearSVR` # Linear support vector regression.

- **`sklearn.tree`**  
  - `sklearn.tree.DecisionTreeClassifier` # Classifier for decision tree-based models.
  - `sklearn.tree.DecisionTreeRegressor` # Regressor for decision tree-based models.

- **`sklearn.ensemble`**  
  - `sklearn.ensemble.RandomForestClassifier` # Random Forest classifier.
  - `sklearn.ensemble.RandomForestRegressor` # Random Forest regressor.
  - `sklearn.ensemble.GradientBoostingClassifier` # Gradient Boosting classifier.
  - `sklearn.ensemble.GradientBoostingRegressor` # Gradient Boosting regressor.
  - `sklearn.ensemble.VotingClassifier` # Ensemble classifier for combining multiple models.
  - `sklearn.ensemble.AdaBoostClassifier` # AdaBoost classifier.
  - `sklearn.ensemble.BaggingClassifier` # Bagging classifier.

- **`sklearn.naive_bayes`**  
  - `sklearn.naive_bayes.GaussianNB` # Gaussian Naive Bayes.
  - `sklearn.naive_bayes.MultinomialNB` # Multinomial Naive Bayes.
  - `sklearn.naive_bayes.ComplementNB` # Complement Naive Bayes.

- **`sklearn.neighbors`**  
  - `sklearn.neighbors.KNeighborsClassifier` # K-nearest neighbors classifier.
  - `sklearn.neighbors.KNeighborsRegressor` # K-nearest neighbors regressor.
  - `sklearn.neighbors.NearestNeighbors` # Nearest neighbors search.

- **`sklearn.cluster`**  
  - `sklearn.cluster.KMeans` # K-means clustering.
  - `sklearn.cluster.DBSCAN` # DBSCAN clustering.
  - `sklearn.cluster.AgglomerativeClustering` # Agglomerative clustering.

- **`sklearn.mixture`**  
  - `sklearn.mixture.GaussianMixture` # Gaussian mixture model.
  - `sklearn.mixture.BayesianGaussianMixture` # Bayesian Gaussian mixture model.

- **`sklearn.metrics`**  
  - `sklearn.metrics.accuracy_score` # Accuracy score.
  - `sklearn.metrics.precision_score` # Precision score.
  - `sklearn.metrics.recall_score` # Recall score.
  - `sklearn.metrics.f1_score` # F1 score.
  - `sklearn.metrics.roc_auc_score` # Area under the ROC curve.
  - `sklearn.metrics.mean_squared_error` # Mean squared error.
  - `sklearn.metrics.r2_score` # R squared score.
  - `sklearn.metrics.silhouette_score` # Silhouette score.

- **`sklearn.pipeline`**  
  - `sklearn.pipeline.Pipeline` # Chain of transformers and estimator.
  - `sklearn.pipeline.FeatureUnion` # Combine features from multiple feature extraction methods.

- **`sklearn.neural_network`**  
  - `sklearn.neural_network.MLPClassifier` # Multi-layer perceptron classifier.
  - `sklearn.neural_network.MLPRegressor` # Multi-layer perceptron regressor.

- **`sklearn.externals`**  
  - `sklearn.externals.joblib` # Lightweight pipelining for data and model storage.

- **`sklearn.impute`**  
  - `sklearn.impute.SimpleImputer` # Impute missing values.
  - `sklearn.impute.KNNImputer` # Impute missing values using k-Nearest Neighbors.

- **`sklearn.utils`**  
  - `sklearn.utils.check_array` # Check if an array-like object is valid.
  - `sklearn.utils.safe_indexing` # Safe indexing for arrays.

- **`sklearn.compose`**  
  - `sklearn.compose.ColumnTransformer` # Transform columns of a dataframe.
  - `sklearn.compose.TransformedTargetRegressor` # Apply transformations to target variable.

- **`sklearn.calibration`**  
  - `sklearn.calibration.CalibratedClassifierCV` # Calibrate classifier outputs.

- **`sklearn.feature_extraction`**  
  - `sklearn.feature_extraction.CountVectorizer` # Convert a collection of text documents to a matrix of token counts.
  - `sklearn.feature_extraction.TfidfVectorizer` # Convert a collection of text documents to a matrix of TF-IDF features.
  - `sklearn.feature_extraction.image.ImageGrid` # Extract features from image data.

- **`sklearn.manifold`**  
  - `sklearn.manifold.LocallyLinearEmbedding` # Locally linear embedding.
  - `sklearn.manifold.Isomap` # Isometric feature mapping.
  - `sklearn.manifold.TSNE` # t-Distributed Stochastic Neighbor Embedding.

- **`sklearn.cross_decomposition`**  
  - `sklearn.cross_decomposition.CCA` # Canonical correlation analysis.
  - `sklearn.cross_decomposition.PLSCanonical` # Partial Least Squares canonical regression.

- **`sklearn.experimental`**  
  - `sklearn.experimental.HistGradientBoostingClassifier` # Histogram-based gradient boosting classifier (experimental).
  - `sklearn.experimental.HistGradientBoostingRegressor` # Histogram-based gradient boosting regressor (experimental).

---
