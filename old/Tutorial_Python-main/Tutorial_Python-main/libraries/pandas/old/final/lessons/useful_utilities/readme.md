
## 📘 Useful Utilities in Pandas

Pandas offers many utility functions and configuration tools that support tasks like debugging, performance tuning, value replacement, and settings control. These are not tied to specific operations but enhance usability and flexibility.

---

### ⚙️ 1. **Utility Functions**

| Function | Description |
|----------|-------------|
| `pd.to_numeric()` | Converts values to numeric, coercing errors if needed |
| `pd.to_datetime()` | Converts object/string to datetime |
| `pd.to_timedelta()` | Converts to time delta |
| `pd.infer_freq()` | Infers frequency string from a DatetimeIndex |
| `pd.unique()` | Returns unique values of a Series |
| `pd.isna()` / `pd.notna()` | Checks for missing values |
| `pd.NA` | Generic missing value marker for all data types |
| `pd.factorize()` | Encodes input values as integer labels |
| `pd.get_dummies()` | One-hot encodes categorical variables |
| `pd.qcut()` / `pd.cut()` | Bins data into quantile-based or fixed intervals |

---

### 🛠️ 2. **Options and Settings**

Pandas has a flexible **options system** for controlling display, performance, and behavior.

#### Key tools:

| Tool | Description |
|------|-------------|
| `pd.set_option()` | Set global option (e.g., display max rows) |
| `pd.get_option()` | Get current value of an option |
| `pd.reset_option()` | Reset option to default |
| `pd.describe_option()` | Describe what an option does |
| `pd.option_context()` | Temporary option context manager (useful in `with` blocks) |

#### Common Options:
| Option | Use |
|--------|-----|
| `'display.max_rows'` | Max number of rows to display |
| `'display.float_format'` | Format for floats (e.g., `'%.2f'`) |
| `'mode.chained_assignment'` | Warn/error on chained assignments |

---

### 🔍 3. **Data Inspection Helpers**

| Tool | Description |
|------|-------------|
| `df.info()` | Summary of DataFrame (dtype, memory usage) |
| `df.memory_usage()` | Shows memory usage by column |
| `df.describe()` | Summary statistics |
| `df.sample()` | Random sample of rows |
| `df.head()` / `df.tail()` | First/last rows |

---

### 🔁 4. **Data Copying and Comparison**

| Function | Description |
|----------|-------------|
| `df.copy()` | Creates a deep or shallow copy |
| `df.equals(other)` | Checks if two DataFrames are equal |
| `df.duplicated()` | Boolean mask for duplicate rows |
| `df.drop_duplicates()` | Removes duplicates |

---

### 🧹 5. **Cleaning Helpers**

| Tool | Description |
|------|-------------|
| `df.fillna()` | Replace missing values |
| `df.replace()` | Replace values using dictionary or rules |
| `df.where()` / `df.mask()` | Conditional replacement |
| `df.clip()` | Limit values to bounds |

---

### 🔍 6. **Performance and Profiling**

| Tool | Description |
|------|-------------|
| `%timeit` | Jupyter magic to measure execution time |
| `df.memory_usage(deep=True)` | Deep memory profiling |
| `df.astype()` | Change dtypes for memory efficiency |

---

### 📂 7. **File and Path Utilities**

| Function | Description |
|----------|-------------|
| `df.to_pickle()` / `pd.read_pickle()` | Fast serialization of objects |
| `df.to_hdf()` / `pd.read_hdf()` | HDF5 support |
| `Path()` (from `pathlib`) | Cleaner path handling with pandas I/O |

---

### 🧪 8. **Testing Utilities**

Pandas includes helpers for writing unit tests:

| Function | Description |
|----------|-------------|
| `pd.testing.assert_frame_equal()` | Check two DataFrames are equal |
| `pd.testing.assert_series_equal()` | Compare two Series |
| `pd.util.testing` | Legacy testing module (deprecated) |

---

### ✅ Summary

| Category | Tools |
|----------|-------|
| Type Conversion | `to_numeric`, `to_datetime`, `to_timedelta` |
| Missing Values | `isna`, `notna`, `NA`, `fillna`, `dropna` |
| Data Inspection | `info`, `describe`, `memory_usage`, `sample` |
| Option Management | `set_option`, `get_option`, `option_context` |
| Cleanup & Comparison | `copy`, `equals`, `duplicated`, `replace` |
| Performance | `memory_usage(deep=True)`, `astype()` |
| Testing | `assert_frame_equal`, `assert_series_equal` |

---
