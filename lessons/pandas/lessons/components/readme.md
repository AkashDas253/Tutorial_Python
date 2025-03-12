## Components of Pandas

---

### **Main Pandas Components**
| **Module/Submodule Name** | **Component**                   | **Usage**                                                                                     |  
|----------------------------|----------------------------------|----------------------------------------------------------------------------------------------|  
| `pandas`                  | `Series`                       | 1D labeled array, supports operations like slicing, arithmetic, and broadcasting.            |  
|                            | `DataFrame`                    | 2D labeled data structure; allows easy data manipulation and analysis.                       |  
|                            | `Index`                        | Immutable sequence object to label axes of Series/DataFrame.                                 |  
|                            | `Categorical`                  | Efficient storage and manipulation of categorical data.                                       |  
|                            | `DatetimeIndex`                | Index for date-time data, supporting time-based filtering and indexing.                      |  
|                            | `PeriodIndex`                  | Index for periods (e.g., monthly, yearly).                                                   |  
|                            | `TimedeltaIndex`               | Index for time deltas or durations.                                                          |  
|                            | `IntervalIndex`                | Index for interval-based data (e.g., numeric ranges).                                        |  
|                            | `MultiIndex`                   | Hierarchical indexing, allowing multi-level labels for rows or columns.                      |  

---

### **Pandas Submodules and Components**
#### **1. pandas.core**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `groupby`                  | Aggregations, transformations, and filtering on groups of data.                              |  
| `window`                   | Rolling and expanding window calculations for time-series analysis.                          |  
| `resample`                 | Down-sampling or up-sampling time-series data to different frequencies.                      |  
| `reshape`                  | Functions to pivot, unstack, stack, or melt data.                                            |  
| `arrays`                   | Specialized array types like `IntegerArray`, `BooleanArray`, and `StringArray`.              |  

---

#### **2. pandas.io**  
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `read_csv` / `to_csv`      | Read/write CSV files for flat data.                                                          |  
| `read_excel` / `to_excel`  | Read/write Excel files (XLS, XLSX).                                                          |  
| `read_sql` / `to_sql`      | Read/write data to/from SQL databases.                                                       |  
| `read_json` / `to_json`    | Read/write JSON files for semi-structured data.                                              |  
| `read_parquet` / `to_parquet` | Read/write Parquet files for optimized columnar data storage.                              |  
| `read_hdf` / `to_hdf`      | Read/write HDF5 files for hierarchical data storage.                                         |  
| `read_clipboard` / `to_clipboard` | Read/write data directly to/from the system clipboard.                               |  
| `read_html` / `to_html`    | Read/write HTML tables.                                                                      |  
| `read_pickle` / `to_pickle`| Serialize and deserialize Python objects to/from Pickle files.                               |  
| `read_feather` / `to_feather` | Read/write Feather files for high-speed binary data.                                      |  

---

#### **3. pandas.plotting**  
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `scatter_matrix`           | Visualize pairwise relationships between features in a DataFrame.                            |  
| `bootstrap_plot`           | Generate bootstrap estimates of statistical parameters.                                      |  
| `parallel_coordinates`     | Plot multivariate data in parallel coordinates.                                              |  
| `lag_plot`                 | Plot lag correlations for time-series data.                                                 |  
| `autocorrelation_plot`     | Plot autocorrelations for time-series data.                                                 |  
| `table`                    | Render DataFrame as a table in Matplotlib plots.                                             |  

---

#### **4. pandas.testing**  
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `assert_frame_equal`       | Test for equality between two DataFrames.                                                    |  
| `assert_series_equal`      | Test for equality between two Series.                                                        |  
| `assert_index_equal`       | Test for equality between two Index objects.                                                 |  

---

#### **5. pandas.util**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `hash_pandas_object`       | Generate a hash for pandas objects for fast comparison.                                       |  
| `testing`                  | Debugging and testing utilities for development.                                              |  
| `unique`                   | Identify unique values in a Series or DataFrame.                                             |  

---

#### **6. pandas.errors**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `EmptyDataError`           | Raised when attempting to read an empty file.                                                |  
| `MergeError`               | Raised during issues with merging/joining data.                                              |  
| `PerformanceWarning`       | Warn about operations with potential performance issues.                                      |  

---

#### **7. pandas.tseries**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `offsets`                  | Define custom time intervals (e.g., `Day`, `MonthEnd`, `QuarterBegin`).                       |  
| `frequencies`              | Handle frequency conversion in time-series data.                                             |  
| `holiday`                  | Utilities to define and work with holiday calendars.                                         |  

---

#### **8. pandas.compat**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `is_platform_windows`      | Check if the current system platform is Windows.                                              |  
| `StringIO`                 | Compatibility for handling string-based file-like objects.                                   |  

---

### **Advanced Features**
| **Component**             | **Usage**                                                                                     |  
|----------------------------|----------------------------------------------------------------------------------------------|  
| `eval`                     | Evaluate Python expressions within a DataFrame for faster operations.                        |  
| `query`                    | Query data with SQL-like syntax within a DataFrame.                                          |  
| `pipe`                     | Method chaining for applying custom functions.                                               |  
| `style`                    | Apply conditional formatting to DataFrames.                                                  |  
| `accessor`                 | Custom `.str`, `.dt`, and `.cat` accessors for strings, datetime, and categorical data.       |  

---

This table captures **all major modules, submodules, and functionalities** within Pandas. Let me know if you want detailed examples for any specific component!

---

Here’s a comprehensive list of all the **Pandas** modules and submodules at the top level, with all related methods grouped under them, as you requested:

---

### **`pandas` Modules and Submodules**

- **`pandas`**
  - `pandas.DataFrame` # Two-dimensional, size-mutable, and potentially heterogeneous tabular data structure.
  - `pandas.Series` # One-dimensional labeled array capable of holding any data type.
  - `pandas.Index` # Immutable array of labels for axes.
  - `pandas.Categorical` # Data structure for categorical data.
  - `pandas.Timestamp` # Represents a single point in time.
  - `pandas.Timedelta` # Represents a difference in time.
  - `pandas.Panel` # (Deprecated) Three-dimensional data structure for labeled data.

- **`pandas.api`**
  - `pandas.api.types` # Functions to check types and perform type casting for pandas objects.
  - `pandas.api.extensions` # Register custom data types and work with extension types in pandas.

- **`pandas.io`**
  - `pandas.io.csv` # Functions for reading and writing CSV files.
  - `pandas.io.excel` # Functions for reading and writing Excel files.
  - `pandas.io.json` # Functions for reading and writing JSON files.
  - `pandas.io.sql` # Functions for interacting with SQL databases.
  - `pandas.io.clipboard` # Functions for reading and writing to clipboard.
  - `pandas.io.parquet` # Functions for reading and writing Parquet files.
  - `pandas.io.feather` # Functions for reading and writing Feather files (efficient columnar storage format).
  - `pandas.io.stata` # Functions for reading and writing Stata files.
  - `pandas.io.msgpack` # Functions for reading and writing Msgpack files (deprecated).
  - `pandas.io.html` # Functions for reading HTML tables into DataFrames.

- **`pandas.util`**
  - `pandas.util.testing` # Testing utilities for internal and external tests (often used with pytest).
  - `pandas.util._decorators` # Decorators for internal use in pandas.
  - `pandas.util._move` # Utility functions for object management, mainly used internally.

- **`pandas.merge`**
  - `pandas.merge` # Function to merge two DataFrames using a database-style join.
  - `pandas.merge_asof` # Merge on the nearest key rather than exact matches.
  - `pandas.merge_ordered` # Merge ordered data while keeping order intact.

- **`pandas.concat`**
  - `pandas.concat` # Concatenates DataFrames along a particular axis.
  - `pandas.concat` with `ignore_index` # Concatenate DataFrames with automatic index reordering.

- **`pandas.pivot`**
  - `pandas.pivot` # Reshapes data by pivoting a DataFrame.
  - `pandas.pivot_table` # Creates a pivot table from DataFrame data.
  - `pandas.melt` # Reshapes data by melting a DataFrame.

- **`pandas.groupby`**
  - `pandas.groupby` # Groups data based on a column for aggregation.
  - `pandas.groupby.agg` # Aggregates data with multiple aggregation operations.
  - `pandas.groupby.transform` # Applies a function to each group and transforms the data.

- **`pandas.cut`**
  - `pandas.cut` # Segments data into discrete bins.
  - `pandas.qcut` # Segments data into equal-sized bins based on quantiles.

- **`pandas.date_range`**
  - `pandas.date_range` # Generates a sequence of dates over a specified period.
  - `pandas.to_datetime` # Converts various types to `Datetime` objects.
  - `pandas.to_timedelta` # Converts data to `Timedelta` objects.

- **`pandas.Indexing`**
  - `pandas.DataFrame.loc` # Label-based indexing for selecting rows/columns.
  - `pandas.DataFrame.iloc` # Position-based indexing for selecting rows/columns.
  - `pandas.DataFrame.at` # Fast label-based scalar access.
  - `pandas.DataFrame.iat` # Fast position-based scalar access.

- **`pandas.apply`**
  - `pandas.DataFrame.apply` # Applies a function along an axis of the DataFrame.
  - `pandas.Series.apply` # Applies a function element-wise on a Series.

- **`pandas.fillna`**
  - `pandas.DataFrame.fillna` # Fills missing values with a specified value or method.
  - `pandas.Series.fillna` # Fills missing values in a Series.

- **`pandas.dropna`**
  - `pandas.DataFrame.dropna` # Removes missing values (NA) from DataFrame.
  - `pandas.Series.dropna` # Removes missing values (NA) from Series.

- **`pandas.sort_values`**
  - `pandas.DataFrame.sort_values` # Sorts DataFrame rows by specified column(s).
  - `pandas.Series.sort_values` # Sorts Series values in ascending or descending order.

- **`pandas.astype`**
  - `pandas.DataFrame.astype` # Casts columns to a specific type.
  - `pandas.Series.astype` # Casts Series to a specific type.

- **`pandas.plotting`**
  - `pandas.plotting.scatter_matrix` # Creates a matrix of scatter plots.
  - `pandas.plotting.density` # Creates density plots.
  - `pandas.plotting.boxplot` # Creates box plots.
  - `pandas.plotting.parallel_coordinates` # Creates a parallel coordinates plot.
  - `pandas.plotting.andrews_curves` # Creates Andrews curves plot.

- **`pandas.tseries`**
  - `pandas.tseries.offsets` # Date offset classes like `Day`, `MonthEnd`, `QuarterEnd`, etc.
  - `pandas.tseries.tools` # Utilities for working with time series, such as conversion functions.

---
