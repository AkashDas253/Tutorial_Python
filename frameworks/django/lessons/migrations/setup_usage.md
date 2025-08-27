# Migration in Django - Setup & Usage

---

## Setup

### Default Setup

* **Migration System Enabled** by default when a Django project is created.
* **Migration Directory**: Each app has a `migrations/` folder (with `__init__.py`) created automatically when migrations are first made.

### Database Setup for Migrations

* Migrations require a database configured in `settings.py`. Example:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # Database backend
        'NAME': BASE_DIR / 'db.sqlite3',        # Database name/path
    }
}
```

* Supported Backends: PostgreSQL, MySQL, SQLite, Oracle.
* Migrations will generate database-specific SQL depending on the backend.

### Initial Migration Setup

* Run for each app when first models are defined:

```bash
python manage.py makemigrations app_name
python manage.py migrate app_name
```

* Creates `0001_initial.py` containing tables for all defined models.

---

## Usage

### Common Workflow

1. **Modify Models** → Add/Change/Delete fields, relationships, constraints.
2. **Generate Migration File** →

   ```bash
   python manage.py makemigrations
   ```

   * Creates a new migration file (`0002_auto_...py`).
3. **Apply Migration to Database** →

   ```bash
   python manage.py migrate
   ```

   * Executes migration operations and updates schema.

---

### Migration Commands (Complete)

| Command                                             | Usage                                                    |
| --------------------------------------------------- | -------------------------------------------------------- |
| `makemigrations`                                    | Creates migration files for model changes.               |
| `migrate`                                           | Applies migrations to database schema.                   |
| `showmigrations`                                    | Lists all migrations with applied/unapplied status.      |
| `sqlmigrate app_name migration_number`              | Shows SQL statements for given migration.                |
| `squashmigrations app_name start_number end_number` | Combines multiple migrations into one.                   |
| `flush`                                             | Resets database (drops all data, re-applies migrations). |
| `check`                                             | Checks project for issues without applying migrations.   |
| `dbshell`                                           | Opens database shell for direct SQL queries.             |

---

### Migration Files Structure

* Located inside each app’s `migrations/`.
* Each migration file has:

  * **Dependencies** → Ensures correct order.
  * **Operations** → Actions to apply (`CreateModel`, `AddField`, `AlterField`, `DeleteModel`).

Example (`0002_add_email_field.py`):

```python
from django.db import migrations, models

class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),  # previous migration
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='email',
            field=models.EmailField(max_length=254, null=True),
        ),
    ]
```

---

### Migration Operations (Complete List)

| Operation             | Description                                   |
| --------------------- | --------------------------------------------- |
| `CreateModel`         | Creates a new model (table).                  |
| `DeleteModel`         | Deletes a model (table).                      |
| `AddField`            | Adds a field (column).                        |
| `RemoveField`         | Removes a field.                              |
| `AlterField`          | Changes field properties (type, constraints). |
| `RenameField`         | Renames a field.                              |
| `RenameModel`         | Renames a model (table).                      |
| `AlterModelTable`     | Changes database table name.                  |
| `AlterUniqueTogether` | Updates unique constraints.                   |
| `AlterIndexTogether`  | Updates index constraints.                    |
| `RunSQL`              | Runs custom SQL.                              |
| `RunPython`           | Runs Python code during migration.            |

---

### Migration State Tracking

* Django tracks applied migrations in the **`django_migrations`** database table.
* Ensures schema matches model definitions.
* Prevents reapplying the same migration twice.

---

### Migration Dependencies

* Each migration depends on the last migration in that app.
* Cross-app dependencies are possible (e.g., `ForeignKey` to another app).
* Ensures ordered execution across apps.

---

### Special Features

#### 1. **Autodetection**

* Detects model changes automatically.
* Prompts for ambiguity (e.g., renaming vs. deleting field).

#### 2. **Manual Migrations**

* Create migration without autodetect:

```bash
python manage.py makemigrations --empty app_name
```

* Useful for custom SQL or Python code.

#### 3. **Custom SQL in Migrations**

```python
migrations.RunSQL(
    "UPDATE users SET is_active = TRUE;",
    "UPDATE users SET is_active = FALSE;"  # reverse
)
```

#### 4. **Custom Python Code in Migrations**

```python
def forwards(apps, schema_editor):
    User = apps.get_model("myapp", "User")
    for user in User.objects.all():
        user.is_active = True
        user.save()

migrations.RunPython(forwards)
```

#### 5. **Reversibility**

* Most migration operations are reversible (can be undone).
* `RunSQL` and `RunPython` need explicit reverse instructions.

#### 6. **Squashing Migrations**

* Reduce large chains of migrations:

```bash
python manage.py squashmigrations app_name 0001 0015
```

* Creates a single replacement migration.

#### 7. **Migration Conflicts**

* Can occur if multiple developers create migrations at the same time.
* Django prompts for manual resolution by editing dependencies.

---

### Best Practices

* **Frequent Migrations** → Keep migrations small and manageable.
* **Review Before Apply** → Check generated migration files.
* **Consistent Across Environments** → Ensure migrations run in dev, staging, production.
* **Use Squash** → Clean up long migration history.
* **Avoid Editing Applied Migrations** → Instead, create a new migration.
* **Test with `sqlmigrate`** → Preview SQL queries before execution.

---

### Usage Scenarios

* **Project Initialization** → `makemigrations` + `migrate` creates initial schema.
* **Feature Update** → Adding new fields/models.
* **Data Migration** → Populate/change existing data with `RunPython`.
* **Schema Optimization** → Adding indexes, constraints.
* **Production Deployment** → Apply same migration files across servers for consistency.

---
