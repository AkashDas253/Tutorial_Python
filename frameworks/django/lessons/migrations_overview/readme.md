## Django Migration 

### Purpose

* Sync Django models with the database schema
* Track changes to model structure over time
* Provide version control over DB schema

---

### Key Concepts

| Concept             | Description                                                              |
| ------------------- | ------------------------------------------------------------------------ |
| Migration           | A Python file that describes changes in models to be reflected in the DB |
| makemigrations      | Generates migration files based on model changes                         |
| migrate             | Applies migration files to the actual database                           |
| Showmigrations      | Lists all migrations and their status                                    |
| sqlmigrate          | Shows raw SQL for a migration file                                       |
| fake/fakemigrations | Marks migration as applied without running SQL                           |
| zero                | Rolls back all migrations                                                |

---

### Typical Workflow

```bash
python manage.py makemigrations         # Create new migration(s)
python manage.py migrate                # Apply to DB
python manage.py showmigrations         # See status of all migrations
python manage.py sqlmigrate app 0001    # See SQL for a migration
```

---

### Advanced Migration Commands

| Command                          | Purpose                                       |
| -------------------------------- | --------------------------------------------- |
| `migrate app zero`               | Roll back all migrations for an app           |
| `migrate app 0001`               | Migrate to a specific migration version       |
| `makemigrations --empty appname` | Create a blank migration for manual editing   |
| `migrate --fake`                 | Mark migration as applied (without execution) |
| `migrate --plan`                 | Show what operations will be performed        |

---

### Where It’s Stored

* Migration files are stored in `your_app/migrations/`
* Each migration file has an auto-generated name (e.g., `0001_initial.py`)

---

### Best Practices

* Keep migrations in version control (Git)
* Avoid editing applied migrations manually
* Always test migrations in a staging/dev environment
* Split large schema changes into multiple migrations if possible

---
