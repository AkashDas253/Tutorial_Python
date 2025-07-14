# Pandas Cheatsheet

## 1. Importing Pandas
- `import pandas as pd`

## 2. Creating Data Structures
- `s = pd.Series(data, index=index)`  # Series
- `df = pd.DataFrame(data, columns=['col1', 'col2'])`  # DataFrame

## 3. Reading and Writing Data
- `df = pd.read_csv('file.csv', sep=',', header=0)`  # Read CSV
- `df.to_csv('output.csv', index=False, encoding='utf-8')`  # Write to CSV
- `df = pd.read_excel('file.xlsx', sheet_name='Sheet1')`  # Read Excel
- `df.to_excel('output.xlsx', index=False, sheet_name='Sheet1')`  # Write to Excel
- `from sqlalchemy import create_engine`  # Read from SQL Database
- `engine = create_engine('sqlite:///database.db')`
- `df = pd.read_sql('SELECT * FROM table_name', con=engine)`

## 4. Viewing Data
- `df.head(n)`  # First n rows
- `df.tail(n)`  # Last n rows
- `df.shape`  # (rows, columns)
- `df.info()`  # DataFrame summary
- `df.describe()`  # Summary of numeric columns
- `df.dtypes`  # Data types of each column

## 5. Accessing Data
- `df['column_name']`  # Access column
- `df[['col1', 'col2']]`  # Access multiple columns
- `df.iloc[row_index]`  # Access row by index position
- `df.loc[label]`  # Access row by label

## 6. Filtering Data
- `filtered_df = df[df['column'] > value]`  # Filter rows based on condition
- `filtered_df = df[(df['col1'] > value1) & (df['col2'] < value2)]`  # AND condition
- `filtered_df = df[(df['col1'] < value1) | (df['col2'] > value2)]`  # OR condition

## 7. Adding and Modifying Columns
- `df['new_column'] = value`  # Add column
- `df['column'] += 1`  # Modify column
- `df['new_column'] = df['existing_column'].apply(lambda x: x * 2)`  # Apply function

## 8. Dropping Data
- `df = df.drop('column_name', axis=1)`  # Drop column
- `df = df.drop(index)`  # Drop row
- `df = df.drop_duplicates()`  # Drop duplicates

## 9. Handling Missing Data
- `df.isnull()`  # Check for NaN
- `df.fillna(value, inplace=True)`  # Fill NaN
- `df.dropna(inplace=True)`  # Drop rows with NaN
- `df['column'].interpolate(method='linear', inplace=True)`  # Interpolate missing values

## 10. Group By and Aggregation
- `grouped = df.groupby('column').mean()`  # Group by and calculate mean
- `agg_df = df.groupby('column').agg({'col1': 'mean', 'col2': 'sum'})`  # Aggregation

## 11. Merging and Joining
- `merged_df = pd.merge(df1, df2, on='key_column', how='inner')`  # Merge DataFrames
- `concat_df = pd.concat([df1, df2], axis=0)`  # Concatenate DataFrames

## 12. Sorting Data
- `df_sorted = df.sort_values(by='column', ascending=False)`  # Sort Data

## 13. Pivot Tables
- `pivot = df.pivot_table(values='value_column', index='index_column', columns='column_name', aggfunc='mean')`  # Pivot Table

## 14. Exporting Data
- `df.to_excel('filename.xlsx', index=False)`  # To Excel
- `df.to_json('filename.json', orient='records')`  # To JSON

## 15. Date and Time Functions
- `df['date_column'] = pd.to_datetime(df['date_column'])`  # Convert to datetime
- `df['year'] = df['date_column'].dt.year`  # Extract year
- `df['month'] = df['date_column'].dt.month`  # Extract month
- `df['day'] = df['date_column'].dt.day`  # Extract day
## 16. Renaming Columns
- `df.rename(columns={'old_column_name': 'new_column_name'}, inplace=True)`  # Rename column

## 17. Applying Functions to Columns
- `df['new_column'] = df['existing_column'].apply(function_name)`  # Apply function to column

## 18. Aggregating Data
- `df.groupby('column').agg({'col1': 'sum', 'col2': 'mean'})`  # Aggregate data

## 19. Reshaping Data
- `df.melt(id_vars=['col1', 'col2'], value_vars=['col3', 'col4'], var_name='variable', value_name='value')`  # Reshape data

## 20. Handling Duplicates
- `df.duplicated(subset=['col1', 'col2'])`  # Check for duplicates
- `df.drop_duplicates(subset=['col1', 'col2'], keep='first', inplace=True)`  # Remove duplicates

## 21. Applying Conditions to Data
- `df['new_column'] = np.where(df['column'] > value, 'condition_met', 'condition_not_met')`  # Apply condition to create new column

## 22. Combining DataFrames
- `df_merged = pd.concat([df1, df2], axis=1)`  # Combine DataFrames horizontally
- `df_merged = pd.concat([df1, df2], axis=0)`  # Combine DataFrames vertically

## 23. Handling Categorical Data
- `df['column'] = df['column'].astype('category')`  # Convert column to categorical data type
- `df['column'].cat.categories`  # Get categories of categorical column
- `df['column'].cat.codes`  # Get codes of categorical column

## 24. Handling Text Data
- `df['column'] = df['column'].str.lower()`  # Convert column to lowercase
- `df['column'].str.contains('pattern')`  # Check if column contains a pattern
- `df['column'].str.replace('old_value', 'new_value')`  # Replace values in column

## 25. Handling Time Series Data
- `df['date_column'] = pd.to_datetime(df['date_column'])`  # Convert to datetime
- `df.set_index('date_column', inplace=True)`  # Set date column as index
- `df.resample('D').mean()`  # Resample time series data

## 26. Handling Numerical Data
- `df['column'] = pd.to_numeric(df['column'], errors='coerce')`  # Convert column to numeric data type
- `df['column'].fillna(value, inplace=True)`  # Fill missing values in column

## 27. Handling Boolean Data
- `df['column'] = df['column'].astype(bool)`  # Convert column to boolean data type
- `df['column'] = df['column'].map({True: 'Yes', False: 'No'})`  # Map boolean values to custom labels

## 28. Handling JSON Data
- `df = pd.read_json('file.json')`  # Read JSON file
- `df.to_json('file.json', orient='records')`  # Write DataFrame to JSON file

## 29. Handling Excel Files with Multiple Sheets
- `excel_file = pd.ExcelFile('file.xlsx')`  # Read Excel file with multiple sheets
- `sheet_names = excel_file.sheet_names`  # Get names of all sheets
- `df = excel_file.parse('Sheet1')`  # Read specific sheet

## 30. Handling Large Data Sets
- `df = pd.read_csv('file.csv', chunksize=1000)`  # Read large CSV file in chunks
- `for chunk in df:`  # Process each chunk
    process_chunk(chunk)