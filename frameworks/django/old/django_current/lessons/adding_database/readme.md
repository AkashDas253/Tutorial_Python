## Adding a Database in Django

---

### Purpose

Adding a database allows Django to store, retrieve, and manage data using models via an Object-Relational Mapper (ORM).

---

### When Is It Needed?

* On **new project creation**
* When **switching from SQLite** to another DB (PostgreSQL, MySQL, etc.)
* When **connecting to a remote or production-grade DB**
* When **scaling or separating services**

---

### Supported Databases by Django

| DBMS       | Engine                                 | Dependency         |
| ---------- | -------------------------------------- | ------------------ |
| SQLite     | `django.db.backends.sqlite3`           | Built-in           |
| PostgreSQL | `django.db.backends.postgresql`        | `psycopg2`         |
| MySQL      | `django.db.backends.mysql`             | `mysqlclient`      |
| Oracle     | `django.db.backends.oracle`            | `cx_Oracle`        |
| MariaDB    | `django.db.backends.mysql`             | `mysqlclient`      |
| Others     | Via 3rd-party packages (e.g., MongoDB) | Use `djongo`, etc. |

---

### Step-by-Step Guide (Adding Database)

#### Step 1: Install Required Package

Install a DB driver compatible with your target DBMS.

```bash
pip install psycopg2           # PostgreSQL
pip install mysqlclient        # MySQL/MariaDB
pip install cx_Oracle          # Oracle
```

---

#### Step 2: Configure `settings.py`

Update the `DATABASES` block:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',  # Change as needed
        'NAME': 'mydb',
        'USER': 'dbuser',
        'PASSWORD': 'dbpassword',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

---

#### Step 3: Create the Database Manually

**PostgreSQL:**

```bash
createdb mydb
```

**MySQL:**

```sql
CREATE DATABASE mydb CHARACTER SET UTF8;
```

---

#### Step 4: Apply Migrations

```bash
python manage.py migrate
```

This creates the necessary schema in the new database.

---

#### Step 5: Verify Connection

```bash
python manage.py dbshell
python manage.py check
```

---

### Optional Enhancements

| Setting           | Description                                 |
| ----------------- | ------------------------------------------- |
| `ATOMIC_REQUESTS` | Auto-wrap requests in DB transactions       |
| `CONN_MAX_AGE`    | Persistent DB connections                   |
| `OPTIONS`         | Extra DB-specific options (e.g., `sslmode`) |
| `TEST`            | Test DB overrides                           |

---

### Switching from SQLite to PostgreSQL (Example)

1. Dump data from SQLite:

   ```bash
   python manage.py dumpdata > data.json
   ```

2. Configure new DB in `settings.py`.

3. Run migrations:

   ```bash
   python manage.py migrate
   ```

4. Load data into new DB:

   ```bash
   python manage.py loaddata data.json
   ```

---

### For Remote Databases

Ensure:

* Network access is allowed (security groups, firewalls).
* DB user has proper privileges.
* SSL may be required.

---
