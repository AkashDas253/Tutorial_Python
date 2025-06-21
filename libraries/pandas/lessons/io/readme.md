## Input/Output in Pandas

### 📌 Definition  
Pandas provides a suite of **I/O functions** to read from and write to a variety of file formats such as **CSV, Excel, JSON, SQL, Parquet, HDF5, HTML, XML**, and others. These functions convert between persistent file formats and in-memory Pandas objects like `DataFrame` or `Series`.

---

### 📥 Input Functions

| Function              | Description                         |
|-----------------------|-------------------------------------|
| `read_csv()`          | Read a CSV file                     |
| `read_excel()`        | Read Excel (.xls/.xlsx) file        |
| `read_json()`         | Read JSON format                    |
| `read_html()`         | Read HTML tables                    |
| `read_sql()`          | Read from SQL database              |
| `read_sql_query()`    | Run SQL query and return DataFrame  |
| `read_sql_table()`    | Read entire SQL table               |
| `read_parquet()`      | Read a Parquet file                 |
| `read_hdf()`          | Read from HDF5 file                 |
| `read_pickle()`       | Load serialized object (via pickle) |
| `read_feather()`      | Read Apache Feather format          |
| `read_orc()`          | Read ORC file format                |
| `read_xml()`          | Read XML file or string             |
| `read_clipboard()`    | Read from clipboard (as table)      |

---

### 📤 Output Functions

| Function              | Description                         |
|-----------------------|-------------------------------------|
| `to_csv()`            | Write DataFrame to CSV              |
| `to_excel()`          | Write to Excel file                 |
| `to_json()`           | Convert to JSON                     |
| `to_html()`           | Convert to HTML table               |
| `to_sql()`            | Store in SQL table                  |
| `to_parquet()`        | Write to Parquet file               |
| `to_hdf()`            | Write to HDF5 format                |
| `to_pickle()`         | Serialize to pickle                 |
| `to_feather()`        | Write to Feather format             |
| `to_orc()`            | Write to ORC format                 |
| `to_xml()`            | Export to XML format                |
| `to_clipboard()`      | Copy table to clipboard             |

---

### ⚙️ Common Parameters in I/O Functions

| Parameter         | Purpose                                           |
|-------------------|---------------------------------------------------|
| `sep`, `delimiter`| Field separator for CSV/TSV                      |
| `header`          | Row to use as column names                        |
| `index_col`       | Use column as row labels                          |
| `usecols`         | Subset of columns to read                         |
| `dtype`           | Specify column data types                         |
| `parse_dates`     | Convert date-like strings to datetime             |
| `chunksize`       | Read file in chunks (for large files)             |
| `compression`     | Handle zipped formats (e.g., 'gzip', 'zip')       |
| `na_values`       | Values to recognize as NA                         |
| `encoding`        | Handle file encodings (e.g., 'utf-8', 'latin1')   |
| `engine`          | Backend engine to use (e.g., 'python', 'c')       |
| `sheet_name`      | Specify sheet in Excel                            |
| `storage_options` | Cloud storage or file system options              |

---

### 📂 File Formats Supported

| Format     | Extension(s)         | Key Notes                                      |
|------------|----------------------|------------------------------------------------|
| CSV        | `.csv`               | Plain text, widely used                        |
| Excel      | `.xls`, `.xlsx`      | Needs `openpyxl` or `xlrd`                     |
| JSON       | `.json`              | Flexible, hierarchical                         |
| HTML       | `.html`              | Table extraction requires `lxml` or `html5lib` |
| SQL        | `.db`, `.sqlite`, etc| Requires SQLAlchemy/DBAPI                      |
| Parquet    | `.parquet`           | Binary, columnar, fast                         |
| HDF5       | `.h5`                | Hierarchical, good for large datasets          |
| Pickle     | `.pkl`               | Python-specific, not portable                  |
| XML        | `.xml`               | Tree-structured data format                    |
| ORC        | `.orc`               | Optimized Row Columnar format (big data)       |
| Feather    | `.feather`           | Very fast, good for data interchange           |

---

### 🧪 Sample Usage

#### Read CSV with custom separator and header
```python
df = pd.read_csv('data.tsv', sep='\t', header=0)
```

#### Write DataFrame to Excel
```python
df.to_excel('output.xlsx', sheet_name='Report')
```

#### Read JSON and parse dates
```python
df = pd.read_json('data.json', convert_dates=True)
```

#### Export to compressed CSV
```python
df.to_csv('data.csv.gz', compression='gzip')
```

#### Read large CSV in chunks
```python
for chunk in pd.read_csv('big.csv', chunksize=5000):
    process(chunk)
```

---

### 📦 Notes on Performance

| Optimization Technique         | Benefit                            |
|--------------------------------|-------------------------------------|
| Use `chunksize`               | Handle large files incrementally    |
| Use binary formats (Parquet)  | Faster I/O and smaller file size    |
| Use specific engines (`c`)    | Faster parsing for CSV              |
| Avoid Pickle for portability  | Prefer open formats like JSON/CSV   |
| Use `columns` or `usecols`    | Reduce memory by selective reading  |

---

### 🔄 Round-Trip Compatibility

| Format      | Round-Trip Safe | Notes                              |
|-------------|------------------|------------------------------------|
| CSV         | No               | Type info and index may be lost    |
| Excel       | Partial          | Type loss possible, esp. dates     |
| JSON        | Partial          | Nested structures can be tricky    |
| Parquet     | Yes              | Supports all dtypes, efficient     |
| HDF5        | Yes              | Ideal for large, hierarchical data |

---
