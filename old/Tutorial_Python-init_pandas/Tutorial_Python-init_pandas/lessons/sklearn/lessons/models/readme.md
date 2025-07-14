# **Algorithms Implement in Scikit-Learn**  

Scikit-Learn provides a wide range of **machine learning algorithms**, categorized by their use cases. Below is a structured overview of all major algorithms available in Scikit-Learn.  

---

## **1. Supervised Learning**  
Supervised learning algorithms predict labels using labeled data.  

### **1.1 Classification Algorithms**  
Used when the target variable is categorical.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Logistic Regression** | `LogisticRegression` | Linear model for binary and multi-class classification. |  
| **Support Vector Machines (SVM)** | `SVC` | Finds an optimal hyperplane for classification. |  
| **K-Nearest Neighbors (KNN)** | `KNeighborsClassifier` | Classifies based on the nearest neighbors. |  
| **Decision Tree** | `DecisionTreeClassifier` | Splits data based on feature conditions. |  
| **Random Forest** | `RandomForestClassifier` | Ensemble of decision trees for better generalization. |  
| **Gradient Boosting** | `GradientBoostingClassifier` | Boosted trees model for better accuracy. |  
| **AdaBoost** | `AdaBoostClassifier` | Boosting algorithm combining weak learners. |  
| **XGBoost (External)** | `xgboost.XGBClassifier` | Optimized gradient boosting. |  
| **Naïve Bayes** | `GaussianNB`, `MultinomialNB`, `BernoulliNB` | Probabilistic classifiers based on Bayes’ theorem. |  
| **Linear Discriminant Analysis (LDA)** | `LinearDiscriminantAnalysis` | Reduces dimensionality while preserving class separability. |  
| **Quadratic Discriminant Analysis (QDA)** | `QuadraticDiscriminantAnalysis` | Similar to LDA but with quadratic boundaries. |  

---

### **1.2 Regression Algorithms**  
Used when the target variable is continuous.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Linear Regression** | `LinearRegression` | Fits a linear relationship between features and target. |  
| **Ridge Regression** | `Ridge` | Linear regression with L2 regularization. |  
| **Lasso Regression** | `Lasso` | Linear regression with L1 regularization. |  
| **ElasticNet** | `ElasticNet` | Combination of L1 and L2 regularization. |  
| **Support Vector Regression (SVR)** | `SVR` | SVM applied for regression tasks. |  
| **K-Nearest Neighbors Regression (KNNR)** | `KNeighborsRegressor` | Predicts based on nearest neighbors. |  
| **Decision Tree Regression** | `DecisionTreeRegressor` | Splits data recursively for regression. |  
| **Random Forest Regression** | `RandomForestRegressor` | Ensemble of decision trees. |  
| **Gradient Boosting Regression** | `GradientBoostingRegressor` | Boosted trees model for regression. |  
| **AdaBoost Regression** | `AdaBoostRegressor` | Boosting technique applied to regression. |  
| **XGBoost Regression (External)** | `xgboost.XGBRegressor` | Efficient gradient boosting for regression. |  

---

## **2. Unsupervised Learning**  
Unsupervised learning is used to find patterns in data without labels.  

### **2.1 Clustering Algorithms**  
Used to group data points into clusters.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **K-Means** | `KMeans` | Partitions data into k clusters. |  
| **Agglomerative Clustering** | `AgglomerativeClustering` | Hierarchical clustering technique. |  
| **DBSCAN** | `DBSCAN` | Density-based clustering for irregular clusters. |  
| **Mean Shift** | `MeanShift` | Nonparametric clustering method. |  
| **Spectral Clustering** | `SpectralClustering` | Uses graph theory for clustering. |  
| **Gaussian Mixture Models (GMM)** | `GaussianMixture` | Probabilistic model using Gaussian distributions. |  
| **Birch Clustering** | `Birch` | Efficient clustering for large datasets. |  

---

### **2.2 Dimensionality Reduction Algorithms**  
Used to reduce feature dimensions while preserving variance.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Principal Component Analysis (PCA)** | `PCA` | Reduces dimensionality while maximizing variance. |  
| **Kernel PCA** | `KernelPCA` | PCA with non-linear kernels. |  
| **t-Distributed Stochastic Neighbor Embedding (t-SNE)** | `TSNE` | Projects high-dimensional data into 2D/3D space. |  
| **Linear Discriminant Analysis (LDA)** | `LinearDiscriminantAnalysis` | Reduces dimensions while maximizing class separability. |  

---

## **3. Anomaly Detection**  
Used to detect outliers in data.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Isolation Forest** | `IsolationForest` | Detects anomalies using tree structures. |  
| **Local Outlier Factor (LOF)** | `LocalOutlierFactor` | Identifies outliers based on local density deviation. |  
| **One-Class SVM** | `OneClassSVM` | Learns a decision boundary for normal samples. |  

---

## **4. Feature Selection & Engineering**  
Techniques to improve model performance by selecting the most relevant features.  

| **Technique** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Variance Threshold** | `VarianceThreshold` | Removes low-variance features. |  
| **SelectKBest** | `SelectKBest` | Selects top k features based on a scoring function. |  
| **Recursive Feature Elimination (RFE)** | `RFE` | Recursively removes least important features. |  
| **Principal Component Analysis (PCA)** | `PCA` | Reduces feature dimensions while retaining variance. |  

---

## **5. Model Selection & Evaluation**  
Methods to improve and evaluate model performance.  

| **Technique** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Train-Test Split** | `train_test_split` | Splits data into training and testing sets. |  
| **Cross-Validation** | `cross_val_score` | Evaluates models using k-fold cross-validation. |  
| **Grid Search** | `GridSearchCV` | Finds best hyperparameters using exhaustive search. |  
| **Randomized Search** | `RandomizedSearchCV` | Randomly searches hyperparameters for efficiency. |  

---

## **6. Ensemble Methods**  
Combining multiple models to improve performance.  

| **Algorithm** | **Scikit-Learn Class** | **Description** |  
|--------------|-----------------|-----------------|  
| **Voting Classifier** | `VotingClassifier` | Combines multiple classifiers using majority voting. |  
| **Bagging Classifier** | `BaggingClassifier` | Trains models on random subsets of data. |  
| **Random Forest** | `RandomForestClassifier` | Collection of decision trees with averaging. |  
| **AdaBoost** | `AdaBoostClassifier` | Boosting algorithm that combines weak learners. |  
| **Gradient Boosting** | `GradientBoostingClassifier` | Sequential boosting of weak models. |  
| **Stacking Classifier** | `StackingClassifier` | Combines predictions of multiple classifiers. |  

---

## **Conclusion**  
This structured list of **all major Scikit-Learn algorithms** helps in selecting the right algorithm for a given task. Once the right algorithm is chosen, detailed implementation can follow.