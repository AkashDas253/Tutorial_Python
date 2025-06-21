## **Transformer in Scikit-Learn**  

A **Transformer** in Scikit-Learn is an object used to **modify or preprocess data**. Transformers implement the `fit()` and `transform()` methods (or `fit_transform()` for efficiency). They are commonly used for feature scaling, encoding categorical variables, dimensionality reduction, and feature selection.  

---

## **Transformer API Structure**  
A Transformer must implement:  
- `fit(X, y=None)`: Learns necessary parameters (e.g., mean and variance for scaling).  
- `transform(X)`: Transforms input data using learned parameters.  
- `fit_transform(X, y=None)`: Combines `fit()` and `transform()` for efficiency.  

---

## **Types of Transformers**  

| **Category** | **Purpose** | **Common Transformers** |  
|-------------|------------|------------------------|  
| **Scaling** | Standardizing or normalizing features | `StandardScaler`, `MinMaxScaler`, `RobustScaler` |  
| **Encoding** | Converting categorical to numerical | `OneHotEncoder`, `LabelEncoder` |  
| **Feature Selection** | Selecting important features | `SelectKBest`, `VarianceThreshold` |  
| **Dimensionality Reduction** | Reducing feature dimensions | `PCA`, `LDA`, `TruncatedSVD` |  

---

## **Common Transformers**  

### **1. Scaling Transformers**  
Used to standardize or normalize numerical features.  

#### **StandardScaler (Mean = 0, Variance = 1)**
```python
from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  
```  

#### **MinMaxScaler (Scales between 0 and 1)**
```python
from sklearn.preprocessing import MinMaxScaler  

scaler = MinMaxScaler()  
X_scaled = scaler.fit_transform(X)  
```  

---

### **2. Encoding Transformers**  
Used to convert categorical data into numerical form.  

#### **One-Hot Encoding**  
```python
from sklearn.preprocessing import OneHotEncoder  

encoder = OneHotEncoder()  
X_encoded = encoder.fit_transform(X)  
```  

#### **Label Encoding**  
```python
from sklearn.preprocessing import LabelEncoder  

encoder = LabelEncoder()  
y_encoded = encoder.fit_transform(y)  
```  

---

### **3. Feature Selection Transformers**  
Used to select the most relevant features.  

#### **SelectKBest (Selects k best features based on statistical tests)**
```python
from sklearn.feature_selection import SelectKBest, f_classif  

selector = SelectKBest(score_func=f_classif, k=10)  
X_selected = selector.fit_transform(X, y)  
```  

#### **VarianceThreshold (Removes features with low variance)**
```python
from sklearn.feature_selection import VarianceThreshold  

selector = VarianceThreshold(threshold=0.01)  
X_selected = selector.fit_transform(X)  
```  

---

### **4. Dimensionality Reduction Transformers**  
Used to reduce feature dimensions while preserving information.  

#### **PCA (Principal Component Analysis)**
```python
from sklearn.decomposition import PCA  

pca = PCA(n_components=2)  
X_reduced = pca.fit_transform(X)  
```  

#### **LDA (Linear Discriminant Analysis)**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  

lda = LinearDiscriminantAnalysis(n_components=1)  
X_reduced = lda.fit_transform(X, y)  
```  

---

## **Using Transformers in Pipelines**  
Transformers are often used inside `Pipeline` objects.  

```python
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  

pipeline = Pipeline([  
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier())  
])  

pipeline.fit(X_train, y_train)  
```  

---

## **Creating a Custom Transformer**  
Custom transformers can be built by subclassing `BaseEstimator` and `TransformerMixin`.  

```python
from sklearn.base import BaseEstimator, TransformerMixin  

class CustomScaler(BaseEstimator, TransformerMixin):  
    def fit(self, X, y=None):  
        self.mean_ = X.mean(axis=0)  
        self.std_ = X.std(axis=0)  
        return self  

    def transform(self, X):  
        return (X - self.mean_) / self.std_  
```  

---

## **Conclusion**  
Transformers are essential in Scikit-Learn for preprocessing data before model training. They help in scaling, encoding, feature selection, and dimensionality reduction, making machine learning pipelines more efficient and reliable.