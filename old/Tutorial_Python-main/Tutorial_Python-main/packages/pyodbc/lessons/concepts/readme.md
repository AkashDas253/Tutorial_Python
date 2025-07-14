## pyodbc – Concepts and Subconcepts

### Setup and Connection
- Installing pyodbc  
  - `pip install pyodbc`
- Importing the library  
  - `import pyodbc`
- ODBC Driver configuration  
  - Checking installed drivers: `pyodbc.drivers()`
- Connection strings  
  - DSN-based connection  
  - DSN-less connection  
  - Trusted connection  
  - Username/password authentication  
  - Connection timeout

### Connection Object (`pyodbc.connect`)
- Attributes  
  - `conn.autocommit`  
  - `conn.timeout`
- Methods  
  - `conn.cursor()`  
  - `conn.commit()`  
  - `conn.rollback()`  
  - `conn.close()`

### Cursor Object (`conn.cursor()`)
- Query Execution  
  - `cursor.execute()`  
  - `cursor.executemany()`  
  - `cursor.callproc()`  
- Fetching Results  
  - `cursor.fetchone()`  
  - `cursor.fetchmany(size)`  
  - `cursor.fetchall()`
- Iterating over results  
  - Using `for row in cursor:`
- Result Metadata  
  - `cursor.description`  
  - `cursor.rowcount`

### SQL Operations
- SELECT Queries  
- INSERT Statements  
- UPDATE Statements  
- DELETE Statements  
- Stored Procedure Calls  
- Transactions  
  - Manual transaction control  
  - Using `autocommit=True`

### Parameterization
- Preventing SQL Injection  
- `?` placeholder for parameters  
- Positional parameters in tuples

### Data Types and Conversions
- Automatic data conversion  
- Manual conversion using `str()`, `int()`, etc.  
- Dealing with `datetime`, `Decimal`, `NULL`, etc.

### Error Handling
- `pyodbc.Error`  
- `pyodbc.InterfaceError`  
- `pyodbc.DatabaseError`  
- Try-except blocks  
- Transaction rollback on error

### Advanced Features
- Stored procedures with input/output parameters  
- Working with temporary tables  
- Working with multiple result sets  
- Executing multiple SQL statements

### Connection Pooling
- ODBC driver-level pooling  
- pyodbc does not provide pooling, but works with it  
- Using with external pooling systems (e.g., SQLAlchemy)

### Integration with Other Libraries
- Pandas (`read_sql`, `to_sql`)  
- SQLAlchemy (via ODBC connection string)  
- Django (via ODBC backends)  
- FastAPI/Flask (as part of backend logic)

### Platform and Compatibility
- Works on Windows/Linux/Mac  
- Compatible with SQL Server, MySQL, PostgreSQL, Oracle, etc.  
- Requires appropriate ODBC driver for target DB

### Debugging and Logging
- Print queries for debugging  
- Using `pyodbc.setdecoding()` and `setencoding()` for encoding issues  
- Debugging connection string errors

---
