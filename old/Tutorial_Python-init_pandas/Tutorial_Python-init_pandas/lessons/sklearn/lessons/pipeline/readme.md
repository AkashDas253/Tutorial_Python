## **Pipeline in Scikit-Learn**  

A **Pipeline** in Scikit-Learn is a way to streamline machine learning workflows by chaining together multiple data processing steps, such as preprocessing, transformation, and model training, into a single object.  

---

### **Key Features of Pipelines**  
- Automates sequential preprocessing and modeling.  
- Ensures proper data transformation before model fitting.  
- Reduces redundancy and improves code readability.  
- Prevents data leakage by applying transformations only to training data.  

---

### **Pipeline Structure**  
A pipeline consists of a sequence of steps, where each step (except the last) must be a **transformer** (`fit_transform()`) and the last step is usually an **estimator** (`fit()` and `predict()`).  

```python
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression  

pipeline = Pipeline([  
    ('scaler', StandardScaler()),  
    ('classifier', LogisticRegression())  
])  
pipeline.fit(X_train, y_train)  
y_pred = pipeline.predict(X_test)  
```  

---

### **Components of a Pipeline**  
| **Component** | **Description** | **Examples** |  
|--------------|----------------|--------------|  
| **Preprocessing** | Transforms input data | `StandardScaler`, `OneHotEncoder`, `PCA` |  
| **Feature Selection** | Selects relevant features | `SelectKBest`, `VarianceThreshold` |  
| **Dimensionality Reduction** | Reduces data dimensions | `PCA`, `LDA` |  
| **Estimator (Final Step)** | Machine learning model | `LogisticRegression`, `RandomForestClassifier` |  

---

### **Types of Pipelines**  

#### **1. Standard Pipeline (Preprocessing + Model)**  
```python
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  

pipeline = Pipeline([  
    ('scaler', StandardScaler()),  
    ('classifier', RandomForestClassifier(n_estimators=100))  
])  
pipeline.fit(X_train, y_train)  
```  

#### **2. Pipeline with Feature Selection**  
```python
from sklearn.feature_selection import SelectKBest, f_classif  

pipeline = Pipeline([  
    ('feature_selection', SelectKBest(score_func=f_classif, k=10)),  
    ('classifier', LogisticRegression())  
])  
pipeline.fit(X_train, y_train)  
```  

#### **3. Pipeline with Dimensionality Reduction**  
```python
from sklearn.decomposition import PCA  

pipeline = Pipeline([  
    ('pca', PCA(n_components=2)),  
    ('classifier', LogisticRegression())  
])  
pipeline.fit(X_train, y_train)  
```  

#### **4. Pipeline with Hyperparameter Tuning (Grid Search)**  
Pipelines can be integrated with hyperparameter tuning using `GridSearchCV`.  
```python
from sklearn.model_selection import GridSearchCV  

params = {  
    'classifier__C': [0.1, 1, 10]  
}  

grid_search = GridSearchCV(pipeline, param_grid=params, cv=5)  
grid_search.fit(X_train, y_train)  
```  

---

### **ColumnTransformer: Handling Different Feature Types**  
For mixed numerical and categorical data, `ColumnTransformer` can be used inside a pipeline.  
```python
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import OneHotEncoder  

preprocessor = ColumnTransformer([  
    ('num', StandardScaler(), ['numerical_feature']),  
    ('cat', OneHotEncoder(), ['categorical_feature'])  
])  

pipeline = Pipeline([  
    ('preprocessor', preprocessor),  
    ('classifier', LogisticRegression())  
])  
pipeline.fit(X_train, y_train)  
```  

---

### **Accessing Pipeline Steps**  
- **Get Named Steps**  
  ```python
  pipeline.named_steps['scaler']
  ```  
- **Modify Parameters**  
  ```python
  pipeline.set_params(classifier__C=0.5)
  ```  

---

### **Conclusion**  
Pipelines provide an efficient and structured way to manage end-to-end machine learning workflows, ensuring consistency, automation, and reduced risk of errors.