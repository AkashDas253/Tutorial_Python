
## **Data Transformation in Pandas**

---

### **Overview**

Data transformation involves modifying data to make it more suitable for analysis, modeling, or reporting. Pandas provides numerous functions for transforming and reshaping data, including applying functions, mapping values, renaming columns, changing data types, binning, and encoding categorical data.

---

### **Key Data Transformation Techniques**

#### **1. Apply Functions to Data**

- **Purpose**: Apply a function element-wise to DataFrame or Series.
- **Syntax**:
  ```python
  df.apply(np.sqrt)  # Apply square root to every element
  df['col'].apply(lambda x: x * 2)  # Apply custom function to a column
  df.applymap(str)  # Apply function to every element (DataFrame only)
  ```
- **Example**:
  ```python
  df['A'] = df['A'].apply(lambda x: x ** 2)  # Square each element in column A
  ```

#### **2. Mapping and Replacing Values**

- **Purpose**: Map or replace values within a column.
- **Syntax**:
  ```python
  df['grade'].map({'A': 4, 'B': 3})  # Map grades to numeric values
  df['col'].replace([1, 2], [10, 20])  # Replace values in a column
  ```
- **Example**:
  ```python
  df['A'] = df['A'].replace([1, 2], [10, 20])  # Replace 1 with 10 and 2 with 20
  ```

#### **3. Renaming Columns or Index**

- **Purpose**: Rename columns or index labels.
- **Syntax**:
  ```python
  df.rename(columns={'old': 'new'}, inplace=True)  # Rename a column
  df.rename(index={0: 'row1'}, inplace=True)  # Rename an index label
  df.columns = df.columns.str.upper()  # Bulk rename using string methods
  ```
- **Example**:
  ```python
  df.rename(columns={'A': 'Column1'}, inplace=True)  # Rename 'A' to 'Column1'
  ```

#### **4. Sorting**

- **Purpose**: Sort data by values in columns or index.
- **Syntax**:
  ```python
  df.sort_values(by='col')  # Sort by a single column
  df.sort_values(by=['col1', 'col2'], ascending=[True, False])  # Multi-column sort
  df.sort_index()  # Sort by index
  ```
- **Example**:
  ```python
  df = df.sort_values(by='A', ascending=False)  # Sort by 'A' in descending order
  ```

#### **5. Changing Data Types**

- **Purpose**: Convert data types of columns.
- **Syntax**:
  ```python
  df['col'].astype('int')  # Convert column to int
  df.astype({'A': 'float', 'B': 'str'})  # Multiple column conversion
  ```
- **Example**:
  ```python
  df['A'] = df['A'].astype(float)  # Convert 'A' to float
  ```

#### **6. Discretization / Binning**

- **Purpose**: Convert continuous data into discrete bins.
- **Syntax**:
  ```python
  pd.cut(df['col'], bins=3)  # Equal-width binning
  pd.cut(df['col'], bins=[0, 5, 10], labels=['Low', 'High'])  # Custom bins
  pd.qcut(df['col'], q=4)  # Quantile-based binning
  ```
- **Example**:
  ```python
  df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 40, 50, 60], labels=['Teen', 'Young Adult', 'Adult', 'Middle-Aged', 'Senior'])
  ```

#### **7. One-Hot Encoding (Dummies)**

- **Purpose**: Convert categorical variables into binary (dummy) variables.
- **Syntax**:
  ```python
  pd.get_dummies(df['category'])  # One-hot encode a single column
  pd.get_dummies(df, columns=['cat_col'])  # One-hot encode multiple columns
  ```
- **Example**:
  ```python
  df_encoded = pd.get_dummies(df['Category'])  # One-hot encode the 'Category' column
  ```

#### **8. Function Pipelines (Chaining)**

- **Purpose**: Chain multiple transformations in a single line of code.
- **Syntax**:
  ```python
  (df.assign(new_col = df['col'] * 2)  # Add new column
     .query('new_col > 10)  # Filter rows where new_col > 10
     .sort_values('new_col')  # Sort by the new column
  )
  ```
- **Example**:
  ```python
  df = (df.assign(new_col = df['A'] * 2)
          .query('new_col > 10')
          .sort_values('new_col')
        )
  ```

#### **9. Log / Exp / Power Transformations**

- **Purpose**: Apply mathematical transformations to data.
- **Syntax**:
  ```python
  np.log1p(df['col'])  # Log(x + 1) transformation
  np.exp(df['col'])  # Exponential transformation
  df['col'] ** 2  # Power transformation (squaring)
  ```
- **Example**:
  ```python
  df['log_col'] = np.log1p(df['A'])  # Apply log(x + 1) to column 'A'
  ```

#### **10. Z-Score Standardization**

- **Purpose**: Standardize data to have a mean of 0 and standard deviation of 1.
- **Syntax**:
  ```python
  (df['col'] - df['col'].mean()) / df['col'].std()  # Z-score normalization
  ```
- **Example**:
  ```python
  df['standardized'] = (df['A'] - df['A'].mean()) / df['A'].std()  # Standardize 'A'
  ```

#### **11. Min-Max Scaling**

- **Purpose**: Scale data to a specific range, typically [0, 1].
- **Syntax**:
  ```python
  (df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())  # Min-Max scaling
  ```
- **Example**:
  ```python
  df['scaled'] = (df['A'] - df['A'].min()) / (df['A'].max() - df['A'].min())  # Min-Max scaling
  ```

#### **12. Aggregation + Transformation (`groupby`)**

- **Purpose**: Group data by a feature and perform aggregation or transformation.
- **Syntax**:
  ```python
  df.groupby('group')['col'].transform('mean')  # Apply aggregation (mean) to groups
  df.groupby('group').transform(lambda x: x - x.mean())  # Group-wise transformation (normalization)
  ```
- **Example**:
  ```python
  df['normalized'] = df.groupby('Category')['Value'].transform(lambda x: (x - x.mean()) / x.std())  # Normalization by group
  ```

---
