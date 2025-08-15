## `SQLAlchemy` in Python

`SQLAlchemy` is a popular **Python SQL toolkit and Object-Relational Mapping (ORM) library** that allows developers to work with relational databases using both **SQL expressions** and **Python objects**. It supports multiple database backends (MySQL, PostgreSQL, SQLite, Oracle, SQL Server, etc.) and provides a unified API.

---

### Overview

* Requires installation:

  ```bash
  pip install sqlalchemy
  ```
* Two primary layers:

  * **Core** – SQL Expression Language for low-level database interaction.
  * **ORM** – Object-oriented layer for mapping Python classes to database tables.
* Fully supports connection pooling, transactions, and schema reflection.

---

### Key Components

#### 1. **Engine** (Database Connection Manager)

* Created using:

  ```python
  from sqlalchemy import create_engine
  engine = create_engine("dialect+driver://user:password@host:port/database", **options)
  ```
* `dialect` – Database type (`mysql`, `postgresql`, `sqlite`, `oracle`, `mssql`).
* `driver` – Database driver (`pymysql`, `psycopg2`, `sqlite`, etc.).
* **Options:**

  * `echo`: Prints SQL statements (`True` or `False`).
  * `pool_size`: Number of connections in the pool.
  * `max_overflow`: Extra connections allowed beyond pool size.
  * `future=True`: Enables SQLAlchemy 2.x style API.

---

#### 2. **Connections & Transactions**

* Connection:

  ```python
  with engine.connect() as conn:
      result = conn.execute(text("SELECT * FROM users"))
  ```
* Transaction control:

  * `commit()` – Commit changes.
  * `rollback()` – Revert changes.

---

#### 3. **SQL Expression Language (Core)**

* Table metadata defined via `Table` objects:

  ```python
  from sqlalchemy import Table, Column, Integer, String, MetaData
  metadata = MetaData()
  users = Table("users", metadata,
      Column("id", Integer, primary_key=True),
      Column("name", String),
  )
  ```
* Query execution:

  ```python
  from sqlalchemy import select
  stmt = select(users).where(users.c.name == "Alice")
  conn.execute(stmt)
  ```

---

#### 4. **ORM (Object Relational Mapping)**

* **Declarative Base** for mapping:

  ```python
  from sqlalchemy.orm import declarative_base
  Base = declarative_base()

  class User(Base):
      __tablename__ = "users"
      id = Column(Integer, primary_key=True)
      name = Column(String)
  ```
* **Session** for persistence:

  ```python
  from sqlalchemy.orm import sessionmaker
  Session = sessionmaker(bind=engine)
  session = Session()
  session.add(User(name="Alice"))
  session.commit()
  ```
* **Queries**:

  ```python
  session.query(User).filter_by(name="Alice").all()
  ```

---

#### 5. **Schema Management**

* `Base.metadata.create_all(engine)` – Create tables.
* `Base.metadata.drop_all(engine)` – Drop tables.

---

#### 6. **Reflection (Reading Existing Database)**

* Automatically load existing tables:

  ```python
  metadata.reflect(bind=engine)
  ```

---

#### 7. **Relationships**

* One-to-many, many-to-many, one-to-one supported via `relationship()`:

  ```python
  from sqlalchemy.orm import relationship
  class Address(Base):
      __tablename__ = "addresses"
      id = Column(Integer, primary_key=True)
      user_id = Column(Integer, ForeignKey("users.id"))
      user = relationship("User", back_populates="addresses")
  ```

---

#### 8. **Type System**

* Common types: `Integer`, `String`, `Text`, `Boolean`, `DateTime`, `Float`, `Numeric`, `LargeBinary`.
* Custom types via `TypeDecorator`.

---

#### 9. **Advanced Features**

* **Connection Pooling** – Automatic pooling for efficiency.
* **Events System** – Hooks for connection, transaction, and ORM events.
* **Hybrid Attributes** – Attributes usable in both Python code and SQL expressions.
* **Eager/Lazy Loading** – Controls how related data is loaded.
* **Migrations** – Done with external tool `Alembic`.

---

#### 10. **Error Handling**

* All exceptions inherit from `sqlalchemy.exc.SQLAlchemyError`.

  * `OperationalError`
  * `IntegrityError`
  * `ProgrammingError`
  * `NoResultFound`
  * `MultipleResultsFound`

---

### Usage Scenarios

* **Core Layer** – When full SQL control is needed.
* **ORM Layer** – When working with Python objects instead of raw SQL.
* **Hybrid Approach** – Combining Core and ORM for flexibility.
* **Cross-Database Apps** – Switching databases without changing logic.
* **Enterprise Applications** – Migrations, connection pooling, complex schemas.

---
