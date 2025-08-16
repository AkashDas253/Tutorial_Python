## `psycopg2` Module in Python

`psycopg2` is the most widely used **PostgreSQL database adapter** for Python. It is written in C for performance and complies with the **Python DB-API 2.0** specification. It supports advanced PostgreSQL features like server-side cursors, asynchronous communication, and large object handling.

---

### Overview

* Requires installation:

  ```bash
  pip install psycopg2       # Compiled binary may require PostgreSQL dev libs  
  pip install psycopg2-binary  # Precompiled wheel (easier install)  
  ```
* Compatible with Python 3.x.
* Supports both **synchronous** and **asynchronous** modes.
* Handles connection pooling via `psycopg2.pool`.

---

### Key Concepts

#### 1. Database Connection

* `psycopg2.connect(dsn=None, **kwargs)` – Opens a connection.
  **Parameters:**

  * `dbname` – Database name.
  * `user` – Username.
  * `password` – Password.
  * `host` – Host address (default `localhost`).
  * `port` – Port number (default `5432`).
  * `dsn` – Data Source Name string: `"dbname=test user=postgres password=secret"`.
  * `connect_timeout` – Timeout in seconds.
  * `sslmode` – SSL mode (`disable`, `require`, `verify-ca`, `verify-full`).

---

#### 2. Cursor Objects

* Created via `connection.cursor(cursor_factory=...)`.
* **Types:**

  * Default – Returns tuples.
  * `DictCursor` – Returns dictionaries.
  * `NamedTupleCursor` – Returns named tuples.
* **Methods:**

  * `execute(sql, params=None)` – Executes a SQL statement.
  * `executemany(sql, seq_of_params)` – Executes for multiple parameter sets.
  * `fetchone()`, `fetchall()`, `fetchmany(size)` – Retrieves query results.

---

#### 3. Transactions

* PostgreSQL automatically starts a transaction.
* `connection.commit()` – Saves changes.
* `connection.rollback()` – Reverts changes.
* `connection.autocommit = True` – Enables autocommit mode.

---

#### 4. Parameter Binding

* Uses `%s` placeholders (not `?` like SQLite):

  ```python
  cur.execute("INSERT INTO users (name, age) VALUES (%s, %s)", ("Alice", 30))
  ```

---

#### 5. Server-Side Cursors (Named Cursors)

* Used for large result sets without loading all rows into memory:

  ```python
  cur = conn.cursor(name="my_cursor")
  cur.execute("SELECT * FROM large_table")
  ```

---

#### 6. Asynchronous Connections

* `psycopg2.connect(..., async_=True)` – Non-blocking operations for event loops.

---

#### 7. Large Object Support

* Access PostgreSQL `BLOB`/`BYTEA` data via `connection.lobject()`.

---

#### 8. Copy Commands (High-Speed Bulk Data Transfer)

* `cursor.copy_from(file, table, sep="\t")` – Load data from file-like object.
* `cursor.copy_to(file, table, sep="\t")` – Export data to file-like object.
* `cursor.copy_expert(sql, file)` – Execute `COPY` with full control.

---

#### 9. Connection Pooling (`psycopg2.pool`)

* `SimpleConnectionPool(minconn, maxconn, **kwargs)` – Basic pooling.
* `ThreadedConnectionPool` – Thread-safe pooling.
* `PersistentConnectionPool` – Keeps persistent connections alive.

---

#### 10. Type Adaptation & Casting

* Automatic mapping between PostgreSQL and Python types.
* Custom type adaptation via `psycopg2.extensions`.

---

#### 11. Error Handling

* All errors are subclasses of `psycopg2.Error`.

  * `OperationalError`
  * `ProgrammingError`
  * `IntegrityError`
  * `DataError`
  * `InterfaceError`
  * `InternalError`
  * `NotSupportedError`

---

#### 12. Context Manager Support

* Both connections and cursors can be used with `with` for automatic closing:

  ```python
  with psycopg2.connect(...) as conn:
      with conn.cursor() as cur:
          cur.execute("SELECT NOW()")
  ```

---

### Usage Scenarios

* Python applications needing fast PostgreSQL connectivity.
* Handling large datasets with server-side cursors.
* High-speed ETL pipelines using `COPY`.
* Web apps with connection pooling for efficiency.
* Asynchronous PostgreSQL queries in event-driven systems.

---
