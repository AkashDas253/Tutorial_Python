## Data Manipulation in Pandas  

Data manipulation in **Pandas** involves selecting, modifying, transforming, filtering, and restructuring data efficiently.

---

### **Selection & Indexing**  

| Method | Description |  
|--------|-------------|  
| `df['col']` | Selects a single column as a Series |  
| `df[['col1', 'col2']]` | Selects multiple columns |  
| `df.loc[row_label, col_label]` | Accesses value by label |  
| `df.iloc[row_index, col_index]` | Accesses value by index |  
| `df.loc[row_label]` | Selects row by label |  
| `df.iloc[row_index]` | Selects row by index |  
| `df.loc[:, 'col']` | Selects a column with all rows |  
| `df.iloc[:, col_index]` | Selects a column by position |  
| `df.set_index('col')` | Sets a column as the index |  
| `df.reset_index()` | Resets index to default integers |  
| `df.reindex(new_index)` | Reindexes using new index values |  

---

### **Data Alignment & Indexing**  

Pandas aligns data based on index labels during operations. If indexes do not match, Pandas fills missing values with `NaN`.  

| Method | Description |  
|--------|-------------|  
| `df.add(df2, fill_value=0)` | Adds with alignment, filling missing values |  
| `df.sub(df2, fill_value=0)` | Subtracts with alignment |  
| `df.mul(df2, fill_value=1)` | Multiplies with alignment |  
| `df.div(df2, fill_value=1)` | Divides with alignment |  
| `df.align(df2, join='outer')` | Aligns two DataFrames based on index |  

---

### **Sorting**  

| Method | Description |  
|--------|-------------|  
| `df.sort_values('col')` | Sorts by column in ascending order |  
| `df.sort_values('col', ascending=False)` | Sorts in descending order |  
| `df.sort_values(['col1', 'col2'])` | Sorts by multiple columns |  
| `df.sort_index()` | Sorts rows by index |  

---

### **Adding & Removing Columns**  

| Method | Description |  
|--------|-------------|  
| `df['new_col'] = values` | Adds a new column |  
| `df.insert(1, 'new_col', values)` | Inserts a column at index `1` |  
| `df.drop('col', axis=1)` | Removes a column |  
| `df.drop(columns=['col1', 'col2'])` | Removes multiple columns |  

---

### **Renaming**  

| Method | Description |  
|--------|-------------|  
| `df.rename(columns={'old': 'new'})` | Renames columns |  
| `df.rename(index={0: 'first'})` | Renames index values |  

---

### **Filtering Data**  

| Method | Description |  
|--------|-------------|  
| `df[df['col'] > n]` | Filters rows where column value > `n` |  
| `df[df['col'] == 'value']` | Filters rows where column = 'value' |  
| `df[(df['col1'] > n) & (df['col2'] < m)]` | Filters based on multiple conditions |  
| `df.query('col > n')` | Queries rows using expressions |  

---

### **Data Transformation**  

| Method | Description |  
|--------|-------------|  
| `df['col'].apply(func)` | Applies a function to a column |  
| `df['col'].map(lambda x: x*2)` | Element-wise transformation |  
| `df.replace({'old': 'new'})` | Replaces values in DataFrame |  
| `df.applymap(func)` | Applies a function to all elements |  

---

### **Handling Missing Data**  

| Method | Description |  
|--------|-------------|  
| `df.isnull()` | Checks for missing values |  
| `df.notnull()` | Checks for non-null values |  
| `df.dropna()` | Drops rows with missing values |  
| `df.dropna(axis=1)` | Drops columns with missing values |  
| `df.fillna(value)` | Replaces missing values |  
| `df.fillna(method='ffill')` | Forward fills missing values |  
| `df.fillna(method='bfill')` | Backward fills missing values |  

---

### **Aggregation & Statistics**  

| Method | Description |  
|--------|-------------|  
| `df.sum()` | Column-wise sum |  
| `df.mean()` | Column-wise mean |  
| `df.median()` | Column-wise median |  
| `df.std()` | Column-wise standard deviation |  
| `df.min(), df.max()` | Column-wise min and max values |  
| `df.count()` | Number of non-null values per column |  
| `df.corr()` | Correlation between numeric columns |  
| `df.cov()` | Covariance between numeric columns |  

---

### **Grouping & Aggregation**  

| Method | Description |  
|--------|-------------|  
| `df.groupby('col').sum()` | Groups by a column and sums values |  
| `df.groupby('col').mean()` | Groups by a column and computes mean |  
| `df.groupby(['col1', 'col2']).count()` | Groups by multiple columns |  
| `df.agg({'col1': 'sum', 'col2': 'mean'})` | Applies multiple aggregations |  

---

### **Reshaping Data**  

| Method | Description |  
|--------|-------------|  
| `df.pivot(index, columns, values)` | Pivots DataFrame |  
| `df.melt(id_vars, var_name, value_name)` | Converts wide format to long format |  
| `df.stack()` | Converts columns into rows |  
| `df.unstack()` | Converts rows into columns |  

---

### **Merging & Joining**  

| Method | Description |  
|--------|-------------|  
| `pd.concat([df1, df2])` | Concatenates along rows |  
| `pd.concat([df1, df2], axis=1)` | Concatenates along columns |  
| `df1.merge(df2, on='col')` | Joins DataFrames on a common column |  
| `df1.join(df2, on='col')` | Joins DataFrames by index |  

---

### **Exporting Data**  

| Method | Description |  
|--------|-------------|  
| `df.to_csv('file.csv')` | Saves as a CSV file |  
| `df.to_excel('file.xlsx')` | Saves as an Excel file |  
| `df.to_json('file.json')` | Saves as a JSON file |  

---

### **Additional Methods**  

| Method | Description |  
|--------|-------------|  
| `df.duplicated()` | Checks for duplicate rows |  
| `df.drop_duplicates()` | Removes duplicate rows |  
| `df.clip(lower, upper)` | Limits values between `lower` and `upper` |  
| `df.nunique()` | Returns number of unique values per column |  
| `df.cumsum()` | Cumulative sum per column |  
| `df.cumprod()` | Cumulative product per column |  
| `df.diff()` | Computes difference between consecutive values |  

---

### **Working with Strings**  

| Method | Description |  
|--------|-------------|  
| `df['col'].str.lower()` | Converts text to lowercase |  
| `df['col'].str.upper()` | Converts text to uppercase |  
| `df['col'].str.strip()` | Removes leading and trailing spaces |  
| `df['col'].str.contains('text')` | Checks if elements contain a substring |  
| `df['col'].str.replace('old', 'new')` | Replaces a substring |  
| `df['col'].str.split('delimiter')` | Splits elements based on a delimiter |  

---

### **Datetime Handling**  
| Method | Description |  
|--------|-------------|  
| `pd.to_datetime(df['col'])` | Converts column to datetime format |  
| `df['col'].dt.year` | Extracts the year |  
| `df['col'].dt.month` | Extracts the month |  
| `df['col'].dt.day` | Extracts the day |  
| `df['col'].dt.weekday` | Extracts the weekday (0=Monday) |  
| `df['col'].dt.strftime('%Y-%m-%d')` | Formats datetime as a string |  
| `df['col'] = df['col'] + pd.Timedelta(days=7)` | Adds 7 days to dates |  

---

### **Window Functions**  
| Method | Description |  
|--------|-------------|  
| `df.rolling(3).mean()` | Computes moving average over 3 rows |  
| `df.expanding().sum()` | Computes cumulative sum |  
| `df.ewm(span=3).mean()` | Computes exponentially weighted moving average |  

---

### **Categorical Data Handling**  
| Method | Description |  
|--------|-------------|  
| `df['col'] = df['col'].astype('category')` | Converts column to categorical type |  
| `df['col'].cat.categories` | Returns category labels |  
| `df['col'].cat.codes` | Returns category codes as integers |  

---

### **Sparse Data Handling**  
| Method | Description |  
|--------|-------------|  
| `df.astype(pd.SparseDtype("float"))` | Converts DataFrame to sparse format to save memory |  
| `pd.SparseDataFrame(df)` | Creates a sparse DataFrame |  

---
