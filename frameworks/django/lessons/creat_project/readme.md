## 🚀 Django Project Creation 

---

### 🔹 1. **Pre-requisites**

| Requirement         | How to Check                              |
| ------------------- | ----------------------------------------- |
| Python 3.8+         | `python --version`                        |
| Django installed    | `pip install django` or `pip show django` |
| Virtual environment | Use `python -m venv venv` (recommended)   |

---

### 🔹 2. **Creating a New Django Project**

#### ✅ Basic Syntax

```bash
django-admin startproject projectname
```

Creates the following:

```
projectname/
├── manage.py
└── projectname/
    ├── __init__.py
    ├── settings.py
    ├── urls.py
    ├── asgi.py
    └── wsgi.py
```

#### ✅ With dot `.` to use current directory

```bash
django-admin startproject projectname .
```

Creates project in the **current directory** (no outer folder).

---

### 🔹 3. **Detailed Breakdown of Commands**

| Command                            | Description                                 |
| ---------------------------------- | ------------------------------------------- |
| `django-admin startproject name`   | Creates a new project named `name`.         |
| `django-admin startproject name .` | Creates a project in the current directory. |
| `python manage.py runserver`       | Runs the development server.                |
| `python manage.py migrate`         | Applies initial DB migrations.              |
| `python manage.py createsuperuser` | Creates admin user for Django admin.        |

---

### 🔹 4. **Post-Project Creation Checklist**

| Step                       | Command / File                                  | Purpose                                  |
| -------------------------- | ----------------------------------------------- | ---------------------------------------- |
| Run server                 | `python manage.py runserver`                    | View default homepage                    |
| Create app                 | `python manage.py startapp appname`             | Add functionality to project             |
| Register app               | `settings.py → INSTALLED_APPS`                  | Add new app to the project               |
| Define routes              | `urls.py`                                       | Wire app views to URLs                   |
| Set templates/static paths | `settings.py` → `TEMPLATES`, `STATICFILES_DIRS` | For frontend integration                 |
| Database setup             | `settings.py → DATABASES`                       | Use SQLite (default) or configure others |
| Environment config         | `.env`, `python-decouple`, or `django-environ`  | For secret key, DB, debug mode etc.      |

---

### 🔹 5. **Project Directory Structure (Post Creation + App)**

```
project_root/
├── manage.py
├── projectname/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── appname/
│   ├── models.py, views.py, etc.
├── templates/
├── static/
├── .env
├── .gitignore
└── requirements.txt
```

---

### 🔹 6. **Best Practices for Creating Django Projects**

| Practice                               | Benefit                         |
| -------------------------------------- | ------------------------------- |
| Use virtual environment (`venv`)       | Isolate dependencies            |
| Use `.env` file                        | Keep sensitive data out of code |
| Create project with `.` if in Git repo | Avoid nested directories        |
| Use meaningful project/app names       | For maintainability             |
| Add `.gitignore` and version control   | Avoid committing unwanted files |

---

### 🔹 7. **Common Errors and Fixes**

| Error Message                                | Solution                                                     |
| -------------------------------------------- | ------------------------------------------------------------ |
| `django-admin: command not found`            | Ensure Django is installed in the environment.               |
| `ImportError: No module named projectname`   | Run commands from the project root where `manage.py` exists. |
| `ModuleNotFoundError: No module named 'app'` | App not added to `INSTALLED_APPS`.                           |
| `PermissionError` on activation (Windows)    | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`    |

---
