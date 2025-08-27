## Migration in Django

### Introduction

Migrations in Django are a way of propagating changes made to **models** (Python code) into the **database schema**. They keep the database schema synchronized with Django models without requiring manual SQL.

---

### Core Concepts

* **Migration**

  * A Python file that records model changes.
  * Created when models.py is modified and `makemigrations` is run.
  * Contains instructions for altering database schema.

* **App Migration Files**

  * Stored in each app’s `migrations/` directory.
  * Numbered sequentially (`0001_initial.py`, `0002_auto.py`, etc.).

* **Migration Operations**

  * Describe changes like creating/deleting models, adding/removing fields, modifying constraints, etc.
  * Examples: `CreateModel`, `AddField`, `AlterField`, `DeleteModel`.

* **Migration State**

  * Django tracks applied migrations in a database table: `django_migrations`.
  * Ensures consistency between code and database.

---

### Workflow

1. **Make Changes in Models**

   * Modify `models.py` (add, change, or delete fields/models).

2. **Generate Migration**

   ```bash
   python manage.py makemigrations
   ```

   * Creates a new migration file inside app’s `migrations/`.

3. **Apply Migration**

   ```bash
   python manage.py migrate
   ```

   * Applies migrations to update the database schema.

4. **Check Migration Status**

   ```bash
   python manage.py showmigrations
   ```

   * Lists applied and unapplied migrations.

5. **Plan Migration**

   ```bash
   python manage.py sqlmigrate app_name migration_number
   ```

   * Shows SQL queries that will be executed.

---

### Migration Management Commands

* `makemigrations` → Create migration files.
* `migrate` → Apply migrations.
* `showmigrations` → List migrations and their status.
* `sqlmigrate` → Display SQL for a migration.
* `squashmigrations` → Combine multiple migrations into one.
* `flush` → Reset database by removing all data and reapplying migrations.

---

### Migration Features

* **Autodetection**

  * Detects changes in models automatically.
  * May require manual edits in complex cases.

* **Dependencies**

  * Migrations can depend on each other across apps.
  * Ensures proper order of execution.

* **Reversible Migrations**

  * Most operations have both forward and backward methods.
  * Example: adding a field can also be undone.

* **Schema Evolution**

  * Handles model evolution over time without dropping/recreating tables.

* **Initial Migrations**

  * First migration (`0001_initial.py`) creates tables for all models.

---

### Best Practices

* Keep migrations small and frequent.
* Always review generated migrations before applying.
* Avoid editing migration files unless necessary.
* Use `squashmigrations` for cleanup in long-lived projects.
* Ensure migrations are consistent across environments (development, staging, production).

---
