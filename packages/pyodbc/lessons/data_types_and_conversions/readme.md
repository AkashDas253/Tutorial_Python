## Data Types and Conversions in `pyodbc`

In `pyodbc`, data types play a crucial role in how SQL queries interact with Python variables. Proper mapping between Python data types and SQL data types ensures smooth interaction between your Python code and the database. `pyodbc` provides automatic type conversion between Python and SQL data types, but understanding the mapping and conversions is essential for efficient query execution and data manipulation.

### Key Concepts

- **Automatic Type Conversion**: `pyodbc` automatically converts between Python data types and SQL data types when executing SQL queries.
- **Data Type Mapping**: `pyodbc` maps SQL data types to corresponding Python data types and vice versa.

### Data Type Mapping

Here are common SQL data types and their corresponding Python data types:

| **SQL Data Type**         | **Python Data Type**       | **Description**                                         |
|---------------------------|----------------------------|---------------------------------------------------------|
| `INTEGER`, `INT`, `BIGINT` | `int`                      | Integer values.                                         |
| `DECIMAL`, `NUMERIC`, `FLOAT`, `REAL` | `float`             | Floating-point numbers.                                |
| `CHAR`, `VARCHAR`, `TEXT`  | `str`                      | String values.                                          |
| `DATE`, `TIME`, `DATETIME`, `TIMESTAMP` | `datetime.date`, `datetime.time`, `datetime.datetime` | Date and time values.                                  |
| `BLOB`, `BINARY`, `VARBINARY` | `bytes`                  | Binary data (byte arrays).                              |
| `BOOLEAN`                  | `bool`                     | Boolean values (True/False).                            |
| `NULL`                     | `None`                     | Null values in SQL are mapped to `None` in Python.       |

### Common Type Conversions

`pyodbc` takes care of many of the conversions automatically. However, you may need to perform some explicit conversions depending on your use case.

#### 1. **SQL to Python Data Type Conversion**

- **Integer to Python `int`**: SQL `INT`, `INTEGER`, `BIGINT` types are converted to Python `int`.
  
  Example:
  ```python
  cursor.execute("SELECT id FROM my_table")
  row = cursor.fetchone()
  id_value = row[0]  # This is an int in Python
  ```

- **Decimal/Floating-Point to Python `float`**: SQL `DECIMAL`, `NUMERIC`, `FLOAT`, `REAL` types are converted to Python `float`.

  Example:
  ```python
  cursor.execute("SELECT price FROM products")
  row = cursor.fetchone()
  price = row[0]  # This is a float in Python
  ```

- **String to Python `str`**: SQL `CHAR`, `VARCHAR`, and `TEXT` types are converted to Python `str`.

  Example:
  ```python
  cursor.execute("SELECT name FROM customers")
  row = cursor.fetchone()
  name = row[0]  # This is a str in Python
  ```

- **Date/Time to Python `datetime.date`, `datetime.time`, `datetime.datetime`**: SQL `DATE`, `TIME`, `DATETIME`, and `TIMESTAMP` types are converted to their respective Python `datetime` objects.

  Example:
  ```python
  cursor.execute("SELECT birth_date FROM employees")
  row = cursor.fetchone()
  birth_date = row[0]  # This is a datetime.date object in Python
  ```

- **Binary Data to Python `bytes`**: SQL `BLOB`, `BINARY`, and `VARBINARY` types are converted to Python `bytes`.

  Example:
  ```python
  cursor.execute("SELECT image_data FROM images")
  row = cursor.fetchone()
  image_data = row[0]  # This is a bytes object in Python
  ```

- **Null to Python `None`**: SQL `NULL` values are converted to Python `None`.

  Example:
  ```python
  cursor.execute("SELECT nullable_column FROM table")
  row = cursor.fetchone()
  value = row[0]  # This will be None if the column value is NULL
  ```

#### 2. **Python to SQL Data Type Conversion**

When inserting or updating data, `pyodbc` automatically converts Python data types to the appropriate SQL data types. However, if needed, you can use explicit casting or conversion functions.

- **Integer to SQL `INT`, `BIGINT`**:
  ```python
  cursor.execute("INSERT INTO my_table (id) VALUES (?)", (123,))
  ```

- **Float to SQL `FLOAT`, `DECIMAL`**:
  ```python
  cursor.execute("INSERT INTO my_table (price) VALUES (?)", (19.99,))
  ```

- **String to SQL `VARCHAR`, `TEXT`**:
  ```python
  cursor.execute("INSERT INTO my_table (name) VALUES (?)", ('John Doe',))
  ```

- **Datetime to SQL `DATETIME`, `TIMESTAMP`**:
  ```python
  from datetime import datetime
  cursor.execute("INSERT INTO my_table (created_at) VALUES (?)", (datetime.now(),))
  ```

- **Bytes to SQL `BLOB`**:
  ```python
  with open('image.png', 'rb') as file:
      image_data = file.read()
      cursor.execute("INSERT INTO images (data) VALUES (?)", (image_data,))
  ```

- **None to SQL `NULL`**:
  ```python
  cursor.execute("INSERT INTO my_table (nullable_column) VALUES (?)", (None,))
  ```

### Handling Data Type Mismatches

In some cases, SQL data types may not match perfectly with Python data types. You can manually cast values in such situations:

- **Explicit Casting in SQL**: Use SQL functions like `CAST()` or `CONVERT()` to convert data types within SQL queries.
  ```python
  cursor.execute("SELECT CAST(column AS VARCHAR) FROM my_table")
  ```

- **Using Python’s `str()` or `float()` for Conversion**:
  ```python
  cursor.execute("INSERT INTO my_table (price) VALUES (?)", (str(price),))
  ```

### Summary

`pyodbc` handles most data type conversions automatically between Python and SQL databases. Understanding the key mappings and performing explicit conversions when necessary can help you avoid issues with data integrity. Proper handling of data types ensures smooth interaction with the database, whether you're fetching, inserting, or updating data.