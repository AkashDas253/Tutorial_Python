## Components of `pyodbc` Module

### Module-Level Functions
- `pyodbc.connect(connection_string, **kwargs)`  
  Creates a new database connection.

- `pyodbc.drivers()`  
  Returns a list of installed ODBC drivers.

- `pyodbc.setencoding(encoding='utf-8')`  
  Sets the default string encoding for Unicode communication.

- `pyodbc.setdecoding(sqltype, encoding='utf-8')`  
  Sets decoding behavior for specific SQL data types.

- `pyodbc.pooling`  
  Boolean attribute to enable or disable connection pooling globally.

---

### Connection Object (`pyodbc.Connection`)
- `cursor()`  
  Returns a new cursor object.

- `commit()`  
  Commits the current transaction.

- `rollback()`  
  Rolls back the current transaction.

- `close()`  
  Closes the connection.

- `autocommit`  
  Property to enable/disable autocommit mode.

- `getinfo(info_type)`  
  Returns database-specific information.

- `timeout`  
  Property to set or get query timeout in seconds.

---

### Cursor Object (`pyodbc.Cursor`)
- `execute(sql, params=...)`  
  Executes a SQL statement.

- `executemany(sql, param_seq)`  
  Executes a SQL statement multiple times with different parameters.

- `callproc(procname, params=...)`  
  Calls a stored procedure.

- `fetchone()`  
  Retrieves the next row of a query result.

- `fetchmany(size)`  
  Retrieves a limited set of rows.

- `fetchall()`  
  Retrieves all remaining rows.

- `nextset()`  
  Moves to the next result set from a multi-statement execution.

- `description`  
  Metadata about result columns (tuple of name, type, etc.).

- `rowcount`  
  Number of rows affected by the last operation.

- `close()`  
  Closes the cursor.

---

### Exception Classes
- `pyodbc.Error`  
  Base class for all exceptions.

- `pyodbc.InterfaceError`  
  Raised for errors related to the database interface.

- `pyodbc.DatabaseError`  
  Raised for database-specific errors.

- `pyodbc.OperationalError`, `ProgrammingError`, `IntegrityError`, etc.  
  Other standard DB-API 2.0 exceptions.

---

### Internal ODBC Constants
- `pyodbc.SQL_*`  
  Constants used for specifying SQL data types, options in `setdecoding()`, and `getinfo()`.

---
