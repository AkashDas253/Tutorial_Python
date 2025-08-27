## Django `manage.py` 

---

### Purpose

`manage.py` is a command-line utility in every Django project used to:

* Interact with your project
* Run administrative tasks
* Manage databases, apps, servers, etc.

---

### Key Uses and Commands

| Command                                         | Description                                           |
| ----------------------------------------------- | ----------------------------------------------------- |
| `python manage.py runserver`                    | Starts development server (default: `127.0.0.1:8000`) |
| `python manage.py startapp <name>`              | Creates a new Django app directory                    |
| `python manage.py makemigrations`               | Creates migration files for model changes             |
| `python manage.py migrate`                      | Applies migrations to the database                    |
| `python manage.py createsuperuser`              | Creates an admin/superuser account                    |
| `python manage.py shell`                        | Opens interactive Python shell with Django context    |
| `python manage.py dbshell`                      | Opens the database shell (if available)               |
| `python manage.py check`                        | Checks for project errors                             |
| `python manage.py showmigrations`               | Displays all migrations and their application status  |
| `python manage.py sqlmigrate <app> <migration>` | Shows SQL for a migration                             |
| `python manage.py collectstatic`                | Collects static files into the static root            |
| `python manage.py test`                         | Runs test cases                                       |
| `python manage.py flush`                        | Removes all data from DB and reinitializes            |
| `python manage.py loaddata <file>`              | Loads data from fixtures (JSON, XML, YAML)            |
| `python manage.py dumpdata`                     | Dumps data as JSON (for backup/transfer)              |

---

### Custom Commands

You can add your own commands in:

```bash
your_app/
└── management/
    └── commands/
        └── yourcommand.py
```

---

### Example: Running Dev Server on Custom Port

```bash
python manage.py runserver 8080
```

---

### Example: Applying Only One App’s Migrations

```bash
python manage.py migrate myapp
```

---
