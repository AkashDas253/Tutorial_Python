
# **Pandas Data Types**

---

## **Overview**

Pandas is built on top of NumPy and uses its **data types (`dtypes`)** internally, while also introducing **extension types** to better support missing data, categoricals, and more.

---

## **1. Core Data Types in Pandas**

| Data Type     | Alias in Pandas  | Base Type     | Description                                      | Nullable (via Extension Type) |
|---------------|------------------|---------------|--------------------------------------------------|-------------------------------|
| `int64`       | `int`            | NumPy         | Integer (64-bit)                                 | ✅ `Int64`                    |
| `float64`     | `float`          | NumPy         | Floating-point (64-bit)                          | ✅ `Float64`                  |
| `bool`        | `bool`           | NumPy         | Boolean                                          | ✅ `boolean`                  |
| `object`      | `object`         | Python        | Mixed or string types                            | ✅ (but not native support)   |
| `string`      | `string[python]` | Extension     | Proper string dtype (instead of object)          | ✅                            |
| `datetime64[ns]` | `datetime`   | NumPy         | Timestamps with nanosecond resolution            | ✅                            |
| `timedelta64[ns]`| `timedelta`  | NumPy         | Differences between datetimes                    | ✅                            |
| `category`    | `category`       | Extension     | Finite set of values; memory-efficient           | ✅                            |
| `complex64`/`128`| `complex`     | NumPy         | Complex numbers                                  | ❌                            |

---

## **2. Extension Data Types**

These are pandas-native dtypes used for better **nullable support** and **efficient storage**.

| Extension Dtype  | Description                              | Backed By     | Example Use                                |
|------------------|------------------------------------------|----------------|---------------------------------------------|
| `Int64`          | Nullable 64-bit integers                 | pandas         | `Series([1, 2, pd.NA], dtype="Int64")`      |
| `Float64`        | Nullable float                           | pandas         | `Series([1.1, None], dtype="Float64")`      |
| `boolean`        | Nullable Boolean                         | pandas         | `Series([True, None], dtype="boolean")`     |
| `string`         | Proper String dtype                      | pandas         | `Series(['a', None], dtype="string")`       |
| `CategoricalDtype`| Categorical data                       | pandas         | `Series(["a", "b"], dtype="category")`      |
| `Sparse[int]`    | Efficient sparse storage for integers     | pandas         | `Series([0, 0, 1], dtype="Sparse[int]")`    |
| `IntervalDtype`  | For interval data                        | pandas         | Used in binning (`pd.cut`) results          |
| `PeriodDtype`    | Periods (e.g., yearly, monthly)          | pandas         | `pd.period_range("2020", periods=3, freq="Y")` |
| `DatetimeTZDtype`| Timezone-aware datetime                  | pandas         | `pd.Series(..., dtype="datetime64[ns, UTC]")` |

---

## **3. dtype Access and Conversion**

| Property or Method     | Description                                      | Example                              |
|------------------------|--------------------------------------------------|--------------------------------------|
| `Series.dtype`         | Shows dtype of Series                            | `s.dtype`                            |
| `DataFrame.dtypes`     | Shows dtype of all columns                       | `df.dtypes`                          |
| `astype(dtype)`        | Convert dtype                                    | `s.astype('float')`                 |
| `convert_dtypes()`     | Convert to best dtype                            | `df.convert_dtypes()`                |

---

## **4. Common Type Mappings Between Python and Pandas**

| Python Type      | Typical Pandas dtype         | Nullable Version     |
|------------------|------------------------------|----------------------|
| `int`            | `int64`                      | `Int64`              |
| `float`          | `float64`                    | `Float64`            |
| `bool`           | `bool`                       | `boolean`            |
| `str`            | `object` / `string`          | `string`             |
| `datetime`       | `datetime64[ns]`             | `datetime64[ns]`     |
| `timedelta`      | `timedelta64[ns]`            | `timedelta64[ns]`    |
| `list/dict`      | `object`                     | —                    |

---

## **5. Inferred Dtype Categories (via `Series.inferred_type`)**

| Inferred Type     | Description                                 |
|-------------------|---------------------------------------------|
| `integer`         | Integer numbers                             |
| `floating`        | Floating-point numbers                      |
| `boolean`         | Boolean values                              |
| `string`          | Python strings                              |
| `mixed`           | Mixed types (e.g., int + str)               |
| `datetime`        | Datetime values                             |
| `timedelta`       | Time differences                            |
| `empty`           | Empty series                                |

---

## **6. Data Type Behavior in Operations**

| Operation                    | Behavior (based on dtype)                          |
|------------------------------|----------------------------------------------------|
| Arithmetic (`+`, `-`, etc.)  | Preserves float unless using nullable Ints        |
| Comparisons                  | Works across types if compatible                  |
| Boolean masking              | Works best with `bool` or `boolean`               |
| Missing values               | Only `object`, extension types, or floats allow NA|
| Grouping/Categorization      | `category` is best for performance                |

---
