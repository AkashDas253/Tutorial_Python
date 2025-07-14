
##  Selection in pandas 

---

###  Direct Column Selection Using `[]`

- Selecting a single column using a string label returns a `Series`.  
  ```python
  df['column_name']
  ```
- Selecting multiple columns using a list of string labels returns a `DataFrame`.  
  ```python
  df[['col1', 'col2']]
  ```

---

###  Label-based Selection using `.loc[]`

- Used to access rows and columns by **label**.
- Format: `df.loc[row_labels, column_labels]`
  ```python
  df.loc['row_label']             # One row
  df.loc[:, 'col']               # All rows, one column
  df.loc['row1':'row3', 'A':'C'] # Row and column slices
  ```
- Accepts:
  - Single label
  - List of labels
  - Slice objects
  - Boolean arrays
  - Callable (lambda)

---

###  Position-based Selection using `.iloc[]`

- Used to access rows and columns by **integer position**.
- Format: `df.iloc[row_indices, column_indices]`
  ```python
  df.iloc[0]                  # First row
  df.iloc[:, 1]               # All rows, second column
  df.iloc[1:4, 0:2]           # Subset by range
  df.iloc[[0, 2], [1, 3]]     # Specific positions
  ```
- Accepts:
  - Integer index
  - List of integers
  - Slice objects
  - Boolean arrays
  - Callable (lambda)

---

###  Fast Scalar Access using `.at[]` and `.iat[]`

- `.at[]` is label-based and faster for **single values**.  
  ```python
  df.at['row_label', 'col_label']
  ```
- `.iat[]` is position-based and faster for **single values**.  
  ```python
  df.iat[0, 1]
  ```

---

###  Boolean Indexing

- Used for filtering rows based on conditions.
  ```python
  df[df['col'] > 5]
  df[(df['a'] > 1) & (df['b'] < 3)]
  ```
- Conditions return a boolean Series of the same length as the DataFrame index.

---

###  Querying Rows using `query()`

- SQL-like syntax to filter rows based on column conditions.
  ```python
  df.query('col > 5 and other == "x"')
  ```
- Only works with column names that are valid Python identifiers.
- String expression evaluated internally.

---

###  Selection using `.where()` and `.mask()`

- `.where(cond)` retains values where condition is `True`, replaces others with `NaN`.
- `.mask(cond)` replaces where condition is `True`.
  ```python
  df.where(df['col'] > 5)
  df.mask(df['col'] < 5)
  ```
- Optional `other` to specify replacement values.
- Optional `inplace` and `axis`.

---

###  Functional Selection with Lambda

- Allows selection logic to be passed as a function.
  ```python
  df.loc[lambda d: d['col'] > 5]
  df.filter(items=lambda x: '2020' in x, axis=1)
  ```

---

###  Membership Filtering using `isin()`

- Returns a boolean Series indicating whether each element is in the passed sequence.
  ```python
  df[df['col'].isin(['A', 'B'])]
  df[~df['col'].isin(['X'])]
  ```

---

###  Filtering using `.filter()`

- Filters columns or rows based on items, like, or regex.
  ```python
  df.filter(items=['A', 'B'])            # Exact names
  df.filter(like='2020', axis=1)         # Substring
  df.filter(regex='^x', axis=1)          # Regex
  ```
- Parameters:
  - `items`: list of labels
  - `like`: string
  - `regex`: pattern
  - `axis`: 0 for index, 1 for columns

---

###  Index/Column Slicing

- Slicing directly using `.index` and `.columns`:
  ```python
  df[df.columns[1:4]]
  df.loc[df.index[:5]]
  ```

---

###  Deprecated Selection Methods (❌)

- `df.ix[]`: Used to mix label and integer selection.  
  **Deprecated since v0.20.0**
- `get_value(row, col)` and `set_value(row, col, val)`  
  **Deprecated since v0.21.0**

---

## ✅ Summary Table of Selection Types

| Method       | Type         | Parameters                        | Returns       |
|--------------|--------------|-----------------------------------|----------------|
| `[]`         | Column       | str or list[str]                  | Series/DataFrame |
| `.loc[]`     | Label        | labels, slices, boolean, callable | Series/DataFrame |
| `.iloc[]`    | Position     | int, list[int], slices            | Series/DataFrame |
| `.at[]`      | Label Scalar | row_label, col_label              | Single value |
| `.iat[]`     | Pos. Scalar  | row_idx, col_idx                  | Single value |
| Boolean      | Filter       | boolean Series                    | Filtered DataFrame |
| `query()`    | Filter       | string expression                 | Filtered DataFrame |
| `where()`    | Conditional  | condition                         | Same-shape DataFrame |
| `mask()`     | Conditional  | condition                         | Same-shape DataFrame |
| `isin()`     | Membership   | list, set, array                  | Boolean Series |
| `.filter()`  | Column/Row   | items, like, regex, axis          | Subset DataFrame |
| Lambda       | Functional   | lambda (with `.loc`, `.filter`)   | Flexible selection |

---
