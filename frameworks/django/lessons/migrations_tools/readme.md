## Django Migration Tool 

---

### What Are Migrations?

Migrations are Django’s way of propagating changes made to models (in `models.py`) into the database schema.

---

### Key Commands

| Command                     | Purpose                                           |
| --------------------------- | ------------------------------------------------- |
| `makemigrations`            | Creates migration files based on model changes    |
| `migrate`                   | Applies migrations to the database                |
| `showmigrations`            | Lists all migrations and their application status |
| `sqlmigrate <app> <number>` | Shows SQL for a specific migration                |
| `makemigrations <app>`      | Creates migrations for specific app               |
| `migrate <app> <migration>` | Applies a specific migration                      |
| `migrate <app> zero`        | Reverts all migrations of an app                  |

---

### Migration Files

* Stored in each app's `migrations/` folder
* Auto-named as `0001_initial.py`, `0002_auto_20250725_1130.py`, etc.
* Contains `operations = [ ... ]` like `CreateModel`, `AddField`, etc.

---

### Manual Operations

```python
from django.db import migrations, models

class Migration(migrations.Migration):

    operations = [
        migrations.AddField(
            model_name='book',
            name='published_date',
            field=models.DateField(null=True),
        ),
    ]
```

---

### Dependency Management

Each migration file:

* Inherits from `migrations.Migration`
* Specifies dependencies using the `dependencies` attribute

```python
dependencies = [
    ('app_name', '0001_initial'),
]
```

---

### Fake Migrations

Useful when DB already matches model but Django doesn’t know:

```bash
python manage.py migrate --fake
```

Fake only one:

```bash
python manage.py migrate <app> <migration> --fake
```

---

### Squashing Migrations

Combines multiple migrations into one for optimization:

```bash
python manage.py squashmigrations app_name start_migration end_migration
```

---

### Resetting Migrations

⚠️ Use carefully in development only:

```bash
rm -rf app_name/migrations/
python manage.py makemigrations
python manage.py migrate --fake
```

---

### Best Practices

* Keep migrations in version control
* Run `makemigrations` and `migrate` after every model change
* Avoid manual edits unless confident
* Use `--plan` to preview actions:

  ```bash
  python manage.py migrate --plan
  ```

---
