## `mysql.connector` Module in Python

The `mysql.connector` module is the official MySQL driver provided by Oracle for connecting Python applications to MySQL databases. It is fully compliant with **Python Database API Specification v2.0 (PEP 249)** and supports both **pure Python** and **C Extension** implementations.

---

### Overview

* Requires installation:

  ```bash
  pip install mysql-connector-python
  ```
* Supports MySQL versions 5.x and 8.x.
* Works across multiple platforms.
* Allows synchronous and asynchronous database operations.
* Provides high-level and low-level connection control.

---

### Key Concepts

#### Database Connections

* `mysql.connector.connect(**params)` – Establishes a connection.
  **Common parameters:**

  * `host`: Hostname or IP of MySQL server (default `"127.0.0.1"`).
  * `port`: Port number (default `3306`).
  * `user`: MySQL username.
  * `password`: MySQL password.
  * `database`: Database name to connect to.
  * `auth_plugin`: Authentication plugin (`mysql_native_password`, `caching_sha2_password`, etc.).
  * `charset`: Character set (`utf8mb4` recommended).
  * `connection_timeout`: Connection timeout in seconds.
  * `autocommit`: Enables/disables autocommit mode (default `False`).

#### Cursor Objects

* Created using `connection.cursor(cursor_class)`

  * Default returns tuples.
  * `MySQLCursorDict` returns rows as dictionaries.
  * `MySQLCursorNamedTuple` returns rows as named tuples.
* **Methods:**

  * `execute(sql[, params])` – Executes a SQL statement.
  * `executemany(sql, seq_of_params)` – Executes for multiple sets of parameters.
  * `fetchone()`, `fetchall()`, `fetchmany(size)` – Retrieve query results.

#### Transactions

* Implicitly started with first query in non-autocommit mode.
* `connection.commit()` – Commits transaction.
* `connection.rollback()` – Rolls back transaction.

#### Parameter Binding

* Uses `%s` placeholders:

  ```python
  cursor.execute("INSERT INTO users (name, age) VALUES (%s, %s)", ("Alice", 30))
  ```

#### Prepared Statements

* Can be used for better performance and security:

  ```python
  cursor.execute("SELECT * FROM users WHERE id = %s", (5,))
  ```

#### Type Handling

* Automatically converts between MySQL and Python types.
* `MySQLCursorPrepared` can handle binary and large data efficiently.

#### Error Handling

* All exceptions are subclasses of `mysql.connector.Error`.

  * `InterfaceError`
  * `DatabaseError`
  * `DataError`
  * `OperationalError`
  * `IntegrityError`
  * `ProgrammingError`
  * `NotSupportedError`

#### Connection Pooling

* Built-in pooling support:

  ```python
  from mysql.connector import pooling
  pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **dbconfig)
  conn = pool.get_connection()
  ```

#### Metadata & Utilities

* `cursor.column_names` – List of column names after query.
* `connection.get_server_info()` – Server version.
* `connection.cmd_statistics()` – MySQL server statistics.
* `connection.set_charset_collation()` – Set connection charset and collation.

#### Bulk Operations

* `executemany()` – Efficient batch inserts/updates.
* `LOAD DATA LOCAL INFILE` – High-speed bulk import (requires enabling).

#### Context Manager Support

* Both `Connection` and `Cursor` support `with` statements:

  ```python
  with mysql.connector.connect(**config) as conn:
      with conn.cursor() as cursor:
          cursor.execute("SELECT NOW()")
  ```

#### Asynchronous Support

* Can integrate with `asyncio` using wrappers or async-specific connectors (e.g., `aiomysql`).

---

### Usage Scenarios

* Production-grade connection to MySQL from Python.
* Web applications requiring persistent relational database access.
* Data ETL pipelines connecting to MySQL.
* Secure applications with authentication plugins.
* Scenarios needing connection pooling for efficiency.

---
