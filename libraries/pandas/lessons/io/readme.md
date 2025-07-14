## Input/Output (I/O) in Pandas

Pandas provides a powerful set of tools for reading and writing data across multiple formats. The focus of I/O in Pandas is to **seamlessly integrate structured data from various sources** (files, databases, web, buffers) into a consistent DataFrame structure for analysis and manipulation.

---

### **Goals of Pandas I/O**

* Efficiently **load** structured/semi-structured data into memory.
* Handle **different file formats** and **data sources** uniformly.
* Allow flexible **exporting** for interoperability with other tools/systems.
* Provide **options to control parsing**, formatting, compression, encoding, etc.

---

## **Input Capabilities (Reading)**

### Flat Files (Text, CSV, TSV, Fixed Width)

* Supports:

  * Delimiter control (`CSV`, `TSV`, etc.)
  * Header auto-detection and skipping
  * Selective column loading
  * NA value parsing
  * Compression (gzip, zip, bz2, xz)
  * Unicode support and encoding detection
* Real-world use:

  * Loading logs, metrics, exported spreadsheets, etc.

### Excel Files

* Reads `.xls`, `.xlsx`, `.xlsm`, `.xlsb`
* Supports:

  * Reading multiple sheets at once
  * Date parsing
  * Formatting preservation (optionally)
  * Headers at arbitrary row positions
* Used heavily in business domains with Excel workflows.

### JSON

* Handles nested JSON, record and split formats
* Can normalize deeply nested objects into tabular structure
* Useful for web APIs and semi-structured data

### HTML

* Parses tables from HTML pages using `lxml` or `html5lib`
* Web scraping use-case: pulls structured tables from sites

### Parquet / ORC

* Columnar binary formats designed for big data ecosystems
* Highly efficient for large datasets
* Supports schema and fast compression

### SQL

* Reads from SQL databases via SQLAlchemy/DBAPI
* Converts SQL query or table to DataFrame
* Allows integration with relational databases

### Others

* Clipboard (`read_clipboard`)
* Pickled pandas objects (`read_pickle`)
* Msgpack (deprecated), Feather, HDF5 (`read_hdf`)

---

## 📤 **Output Capabilities (Writing/Exporting)**

### CSV, TXT, TSV

* Most common export format
* Controls for:

  * Delimiters
  * Quoting, escaping
  * Compression
  * Header/index inclusion
  * Encoding (e.g., UTF-8, UTF-16)

### Excel

* Can write multiple DataFrames to separate sheets
* Supports formatting and date handling
* Allows formulas and styling via engines like `openpyxl`, `xlsxwriter`

### JSON

* Supports compact and pretty formats
* Can output in `records`, `split`, `index`, and `table` formats

### HTML

* Exports DataFrames as styled HTML tables
* Used for dashboards or embedding reports

### Parquet / ORC

* Optimized for storage and retrieval performance
* Preferred for data lakes and cloud storage
* Keeps metadata/schema info

### SQL

* Can write to SQL tables directly
* Options to append or replace data
* Good for pipelines involving databases

### Pickle

* Python-native object serialization
* Fast for pandas-native workflows but not language-agnostic

---

## **Key Functionalities and Controls Across I/O**

| Feature                        | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| **Compression support**        | gzip, bz2, zip, xz support for both reading and writing   |
| **Chunked loading**            | Read large files in chunks (`chunksize`) to manage memory |
| **Encoding handling**          | Full control over character encoding during import/export |
| **Selective columns/rows**     | Read only required data (`usecols`, `nrows`, `skiprows`)  |
| **Schema inference**           | Auto or manual data type assignment                       |
| **Datetime parsing**           | Convert strings to datetime objects during load           |
| **Index preservation**         | Choose whether to write DataFrame index or not            |
| **Multi-format compatibility** | Seamlessly switch between formats without data loss       |

---

## Real-World Relevance

* **Data Ingestion**: From operational systems, logs, APIs, spreadsheets
* **ETL Pipelines**: Intermediate loading/exporting in Parquet/CSV
* **Reporting**: Exporting to Excel, HTML, styled reports
* **Data Science**: Save/restore DataFrames using Pickle, HDF5
* **Web/Cloud**: Interface with cloud storage formats and databases

---
