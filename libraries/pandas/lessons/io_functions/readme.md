# **Pandas Data Input / Output (I/O)**

---

### **CSV**

```python
pd.read_csv(
    filepath_or_buffer,       # path, URL, or file-like object
    sep=',',                  # delimiter between fields
    header='infer',           # row to use as column names
    index_col=None,           # column to use as index
    usecols=None,             # which columns to read
    dtype=None,               # data types for columns
    parse_dates=False         # parse datetime columns
)

df.to_csv(
    path_or_buf,              # file path or buffer to write
    sep=',',                  # separator to use
    index=True,               # write row index
    header=True,              # write column names
    encoding=None,            # encoding like 'utf-8'
    compression=None          # e.g., 'gzip', 'zip', etc.
)
```

---

### **Excel**

```python
pd.read_excel(
    io,                       # file path or buffer
    sheet_name=0,             # sheet name or index
    header=0,                 # row for column headers
    index_col=None,           # column to set as index
    usecols=None              # read only specific columns
)

df.to_excel(
    excel_writer,             # file path or ExcelWriter object
    sheet_name='Sheet1',      # name of the sheet
    index=True,               # write index
    header=True               # write column names
)
```

---

### **JSON**

```python
pd.read_json(
    path_or_buf,              # path or buffer
    orient=None,              # expected JSON orientation
    lines=False               # read JSON lines format
)

df.to_json(
    path_or_buf,              # file path or buffer
    orient='records',         # format of JSON structure
    lines=False               # write as JSON lines
)
```

---

### **SQL**

```python
pd.read_sql(
    sql,                      # SQL query or table name
    con,                      # SQLAlchemy or DBAPI connection
    index_col=None,           # index column
    parse_dates=None          # convert columns to datetime
)

df.to_sql(
    name,                     # name of SQL table
    con,                      # connection object
    if_exists='fail',         # 'fail', 'replace', or 'append'
    index=True,               # write DataFrame index
    index_label=None          # name for index column
)
```

---

### **Parquet**

```python
pd.read_parquet(
    path,                     # file path or buffer
    engine='auto'             # 'auto', 'pyarrow', or 'fastparquet'
)

df.to_parquet(
    path,                     # output file path
    engine='auto',            # parquet engine to use
    index=True                # include index in output
)
```

---

### **HDF5**

```python
pd.read_hdf(
    path_or_buf,              # HDF file path or buffer
    key=None                  # dataset name in file
)

df.to_hdf(
    path_or_buf,              # output path or buffer
    key,                      # name for dataset
    mode='a',                 # append, read, or write mode
    format=None               # 'fixed' or 'table'
)
```

---

### **ORC**

```python
pd.read_orc(
    path                      # ORC file path
)

df.to_orc(
    path                      # path to save ORC file
)
```

---

### **HTML**

```python
pd.read_html(
    io,                       # URL or HTML string
    match='.+',               # regex for matching table
    header=0                  # row to use as column names
)

df.to_html(
    buf=None,                 # file path or buffer
    columns=None,             # list of columns to write
    header=True,              # write column names
    index=True                # write row index
)
```

---

### **Clipboard**

```python
pd.read_clipboard()           # reads copied table into DataFrame

df.to_clipboard(
    index=False               # whether to write index
)
```

---

### **Stata**

```python
pd.read_stata(
    filepath_or_buffer        # Stata file to read
)

df.to_stata(
    path,                     # output path
    write_index=True          # include index column
)
```

---

### **SAS**

```python
pd.read_sas(
    filepath_or_buffer,       # SAS file path
    format='sas7bdat'         # file format type
)
```

---

### **Common Parameters Across I/O**

| Parameter        | Description                                                      |
|------------------|------------------------------------------------------------------|
| `encoding`       | Character encoding (e.g., `'utf-8'`, `'latin1'`)                 |
| `compression`    | Compression type (`'gzip'`, `'bz2'`, `'zip'`, `'xz'`, `'infer'`) |
| `index`          | Whether to include the DataFrame index                          |
| `header`         | Include column names in output                                   |
| `chunksize`      | Read/write data in chunks for large files                        |
| `nrows`          | Number of rows to read                                           |
| `skiprows`       | Skip initial rows                                                |
| `usecols`        | Import only specific columns                                     |
| `dtype`          | Explicitly set data types                                        |
| `parse_dates`    | Convert string columns to datetime automatically                 |

---
