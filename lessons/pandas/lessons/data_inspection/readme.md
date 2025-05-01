
## **Data Inspection in Pandas**

Data inspection is the **first step in Exploratory Data Analysis (EDA)**. It helps you understand the structure, size, data types, null values, and overall quality of your dataset.

---

###  **Basic Structure Inspection**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.head(n)`              | View first `n` rows (default 5)                 | `df.head(10)`                     |
| `df.tail(n)`              | View last `n` rows (default 5)                  | `df.tail(10)`                     |
| `df.shape`                | Returns (rows, columns)                         | `df.shape`                        |
| `df.columns`              | Returns column names                            | `df.columns`                      |
| `df.index`                | Returns row index values                        | `df.index`                        |
| `df.dtypes`               | Returns data types of each column               | `df.dtypes`                       |
| `df.size`                 | Total number of elements                        | `df.size`                         |
| `df.ndim`                 | Number of dimensions (1 = Series, 2 = DataFrame)| `df.ndim`                         |

---

###  **Data Type & Memory Info**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.info(verbose=None, memory_usage=None)` | Summary of DataFrame incl. nulls and types | `df.info()`          |
| `df.memory_usage(deep=False)` | Memory usage of each column                | `df.memory_usage(deep=True)`     |

---

###  **Summary Statistics**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.describe(include=None, exclude=None)` | Statistical summary for numeric columns | `df.describe()`     |
| `df.describe(include='all')` | Summary for all columns incl. objects       | `df.describe(include='all')`     |
| `df.value_counts()`       | Count of unique values (Series only)           | `df['col'].value_counts()`       |
| `df['col'].unique()`      | Unique values in a column                      | `df['col'].unique()`             |
| `df['col'].nunique()`     | Number of unique values                        | `df['col'].nunique()`            |

---

###  **Missing Data Inspection**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.isnull()`             | Boolean DataFrame of missing values             | `df.isnull()`                     |
| `df.notnull()`            | Boolean DataFrame of non-missing values         | `df.notnull()`                    |
| `df.isnull().sum()`       | Count of missing values per column              | `df.isnull().sum()`               |
| `df[df.isnull().any(axis=1)]` | Rows with at least one null value         |                                   |

---

###  **Column-Level Inspection**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df['col'].dtype`         | Data type of a column                          | `df['age'].dtype`                 |
| `df['col'].value_counts()`| Frequency count of values                      | `df['gender'].value_counts()`     |
| `df['col'].describe()`    | Summary stats for a single column              | `df['salary'].describe()`         |

---

###  **Sample Inspection**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.sample(n=5)`          | Randomly sample `n` rows                        | `df.sample(5)`                    |
| `df.sample(frac=0.1)`     | Sample 10% of the data                         | `df.sample(frac=0.1)`             |
| `df.sample(n=3, random_state=42)` | Reproducible sample of 3 rows        |                                   |

---

###  **Unique and Frequency Analysis**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df['col'].unique()`      | Unique values                                   | `df['col'].unique()`              |
| `df['col'].nunique()`     | Count of unique values                         | `df['col'].nunique()`             |
| `df['col'].value_counts(normalize=True)` | Relative frequency            |                                   |

---

###  **Index & Column Names Formatting**

| Method                    | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.columns.tolist()`     | List of column names                           | `df.columns.tolist()`             |
| `df.index.tolist()`       | List of index values                          | `df.index.tolist()`               |
| `df.rename(columns={})`   | Rename columns                                 | `df.rename(columns={'A': 'a'})`   |

---

###  **Advanced Introspection**

| Method / Property         | Description                                     | Syntax / Example                  |
|---------------------------|-------------------------------------------------|-----------------------------------|
| `df.select_dtypes(include=[...])` | Filter columns by data type         | `df.select_dtypes(include='number')` |
| `df.columns.str.contains('pattern')` | Filter columns by name pattern  | `df.loc[:, df.columns.str.contains("score")]` |

---

Would you like to continue with:
- **Custom Function Applications (`apply`, `map`, `applymap`, `pipe`)**
- **Categorical Data**
- Or should I combine all notes so far into one download-ready Markdown/PDF file?