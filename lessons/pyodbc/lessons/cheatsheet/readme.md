## `pyodbc` Cheatsheet

### Basic Setup and Connection

- **Installing `pyodbc`**:
  ```bash
  pip install pyodbc
  ```

- **Connecting to a Database**:
  ```python
  import pyodbc
  conn = pyodbc.connect('DSN=DataSource;UID=user;PWD=password')
  cursor = conn.cursor()
  ```

### Connection Management

- **Close Connection**:
  ```python
  conn.close()
  ```

- **Autocommit Mode**:
  ```python
  conn.autocommit = True  # or False for manual commit
  ```

- **Transaction Management**:
  ```python
  conn.commit()  # Commit transaction
  conn.rollback()  # Rollback transaction
  ```

### Cursor Management

- **Create Cursor**:
  ```python
  cursor = conn.cursor()
  ```

- **Execute SQL Command**:
  ```python
  cursor.execute("SELECT * FROM table_name")
  ```

- **Fetching Data**:
  - **Single Row**:
    ```python
    row = cursor.fetchone()
    ```
  - **All Rows**:
    ```python
    rows = cursor.fetchall()
    ```
  - **Batch Fetching**:
    ```python
    rows = cursor.fetchmany(size=10)  # Fetches 10 rows at a time
    ```

- **Fetching Specific Columns**:
  ```python
  cursor.execute("SELECT column1, column2 FROM table_name")
  row = cursor.fetchone()
  print(row.column1, row.column2)
  ```

### Executing SQL Queries

- **Insert Data**:
  ```python
  cursor.execute("INSERT INTO table_name (col1, col2) VALUES (?, ?)", (value1, value2))
  conn.commit()
  ```

- **Update Data**:
  ```python
  cursor.execute("UPDATE table_name SET col1 = ? WHERE col2 = ?", (new_value, condition_value))
  conn.commit()
  ```

- **Delete Data**:
  ```python
  cursor.execute("DELETE FROM table_name WHERE col1 = ?", (value))
  conn.commit()
  ```

- **Using `executemany()` for Batch Inserts**:
  ```python
  data = [('John', 'HR'), ('Jane', 'Engineering')]
  cursor.executemany("INSERT INTO employees (name, department) VALUES (?, ?)", data)
  conn.commit()
  ```

### Fetching Database Information

- **Get Database Info**:
  ```python
  db_info = conn.getinfo(pyodbc.SQL_DBMS_NAME)
  print(db_info)
  ```

- **Get Driver Information**:
  ```python
  driver_name = conn.getinfo(pyodbc.SQL_DRIVER_NAME)
  print(driver_name)
  ```

### Error Handling

- **Try-Except for Errors**:
  ```python
  try:
      conn = pyodbc.connect('DSN=DataSource;UID=user;PWD=password')
  except pyodbc.DatabaseError as e:
      print(f"Error: {e}")
  ```

- **Common Errors**:
  - `pyodbc.InterfaceError`: Connection issues.
  - `pyodbc.DatabaseError`: Errors related to SQL execution.

### Working with Large Data Types

- **Inserting BLOBs (Binary Large Objects)**:
  ```python
  with open('file.png', 'rb') as f:
      data = f.read()
  cursor.execute("INSERT INTO files (file_data) VALUES (?)", (data,))
  conn.commit()
  ```

- **Inserting CLOBs (Character Large Objects)**:
  ```python
  long_text = "Some long text here..."
  cursor.execute("INSERT INTO text_data (description) VALUES (?)", (long_text,))
  conn.commit()
  ```

### Working with Parameters

- **Parameterized Queries**:
  ```python
  cursor.execute("SELECT * FROM table WHERE column = ?", (param_value,))
  ```

- **Parameterized Inserts**:
  ```python
  cursor.execute("INSERT INTO table (column1, column2) VALUES (?, ?)", (value1, value2))
  conn.commit()
  ```

### Connection Pooling (via `SQLAlchemy`)

- **SQLAlchemy Integration for Connection Pooling**:
  ```python
  from sqlalchemy import create_engine
  engine = create_engine('mssql+pyodbc://user:password@dsn')
  with engine.connect() as conn:
      result = conn.execute("SELECT * FROM employees")
      for row in result:
          print(row)
  ```

### Advanced Features

- **Scrollable Cursors**:
  ```python
  cursor.execute("SELECT * FROM employees")
  cursor.scroll(5, mode='absolute')  # Move to the 5th row
  row = cursor.fetchone()
  print(row)
  ```

- **Async Query Execution (via ThreadPoolExecutor)**:
  ```python
  import asyncio
  from concurrent.futures import ThreadPoolExecutor

  def run_query():
      conn = pyodbc.connect('DSN=DataSource;UID=user;PWD=password')
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM large_table")
      return cursor.fetchall()

  async def async_query():
      loop = asyncio.get_event_loop()
      with ThreadPoolExecutor() as pool:
          result = await loop.run_in_executor(pool, run_query)
          print(result)

  asyncio.run(async_query())
  ```

### Best Practices

- **Always Close Connections**:
  ```python
  conn.close()
  ```

- **Use `with` Statement for Automatic Resource Management**:
  ```python
  with pyodbc.connect('DSN=DataSource;UID=user;PWD=password') as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT * FROM table")
  ```

- **Use Parameterized Queries**: Always use parameterized queries to avoid SQL injection attacks and improve performance.

---

This `pyodbc` cheatsheet summarizes key features, functions, and methods for efficiently working with databases via ODBC in Python. It covers connection management, query execution, fetching data, and error handling for seamless database interactions.