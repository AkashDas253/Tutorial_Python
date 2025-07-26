## 📁 Django Project Files

---

### 🔹 1. **Core Project Files (created by `startproject`)**

| File/Folder       | Purpose                                                                            |
| ----------------- | ---------------------------------------------------------------------------------- |
| `manage.py`       | Command-line tool to run and manage the Django project (runserver, migrate, etc.). |
| `projectname/`    | Configuration package for the project. Contains global settings and entry points.  |
| ├── `__init__.py` | Marks the directory as a Python package.                                           |
| ├── `settings.py` | Global settings (DB, middleware, apps, templates, etc.).                           |
| ├── `urls.py`     | Root URL dispatcher (maps URLs to views).                                          |
| ├── `asgi.py`     | ASGI entry point for asynchronous deployment.                                      |
| └── `wsgi.py`     | WSGI entry point for synchronous deployment (used by most servers).                |

---

### 🔹 2. **App-Level Files (created by `startapp`)**

Each app is a self-contained module in `appname/`:

| File/Folder       | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `__init__.py`     | Marks the app directory as a Python package.                   |
| `admin.py`        | Registers models for the Django Admin interface.               |
| `apps.py`         | App configuration and metadata.                                |
| `models.py`       | Contains database models using Django ORM.                     |
| `views.py`        | Contains views (functions or classes to handle HTTP requests). |
| `tests.py`        | Unit tests for the app.                                        |
| `migrations/`     | Stores migration scripts for the app's models.                 |
| └── `__init__.py` | Required for migration discovery.                              |

---

### 🔹 3. **Optional App-Level Files (manually created)**

| File                       | Purpose                                                                 |
| -------------------------- | ----------------------------------------------------------------------- |
| `urls.py`                  | App-specific URL routing.                                               |
| `forms.py`                 | Django Forms classes (for form validation and rendering).               |
| `serializers.py`           | Used in Django REST Framework to convert models to JSON and vice versa. |
| `signals.py`               | Contains signal handlers for lifecycle events (e.g., post\_save).       |
| `permissions.py`           | Custom permission classes for APIs.                                     |
| `filters.py`               | Filtering logic for APIs (often used with DRF).                         |
| `tasks.py`                 | Celery tasks or background job definitions.                             |
| `context_processors.py`    | Custom template context injectors.                                      |
| `utils.py` or `helpers.py` | Utility functions or shared code.                                       |

---

### 🔹 4. **Project-Level Shared Resources**

| Folder          | Purpose                                                             |
| --------------- | ------------------------------------------------------------------- |
| `templates/`    | HTML templates used across apps.                                    |
| `static/`       | Static files (CSS, JS, images) shared across the project.           |
| `media/`        | Uploaded media files (e.g., user profile pictures).                 |
| `locale/`       | Translation files for internationalization (`.po`, `.mo`).          |
| `logs/`         | Centralized logging directory (manual).                             |
| `requirements/` | Split requirements files (e.g., `base.txt`, `dev.txt`, `prod.txt`). |

---

### 🔹 5. **Environment and Configuration Files**

| File                       | Purpose                                                       |
| -------------------------- | ------------------------------------------------------------- |
| `.env`                     | Stores environment variables (SECRET\_KEY, DB creds).         |
| `.gitignore`               | Tells Git which files/folders to ignore.                      |
| `README.md`                | Project documentation.                                        |
| `pyproject.toml`           | Python packaging and tool configuration (e.g., Black, isort). |
| `setup.py`                 | If packaging the project as a Python module.                  |
| `Pipfile` / `Pipfile.lock` | Alternative to `requirements.txt` (used with pipenv).         |
| `requirements.txt`         | Lists Python package dependencies.                            |
| `Makefile`                 | Developer shortcuts for common commands.                      |
| `.editorconfig`            | Coding style rules for editors.                               |
| `tox.ini`                  | Config for testing across environments.                       |
| `pytest.ini`               | Configuration for pytest.                                     |
| `coverage.xml`             | Code coverage report.                                         |

---

### 🔹 6. **Deployment Files**

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `Dockerfile`         | Docker build instructions for the project.       |
| `docker-compose.yml` | Multi-container orchestration.                   |
| `Procfile`           | Entry point definition (used by Heroku).         |
| `gunicorn.conf.py`   | Gunicorn server configuration.                   |
| `supervisord.conf`   | Supervisor configuration for process management. |

---

### 🔹 7. **IDE/Tool Specific Files**

| File/Folder                         | Purpose                               |
| ----------------------------------- | ------------------------------------- |
| `.vscode/`                          | VS Code project-specific settings.    |
| `.idea/`                            | JetBrains (PyCharm) project settings. |
| `.pylintrc`, `.flake8`, `.mypy.ini` | Linter and type-checker configs.      |

---

### 🔹 8. **Runtime/Generated Files (Not Version Controlled)**

| File/Folder      | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| `__pycache__/`   | Compiled bytecode by Python interpreter.        |
| `*.pyc`          | Bytecode files.                                 |
| `.pytest_cache/` | Pytest runtime cache.                           |
| `.mypy_cache/`   | mypy type-checker cache.                        |
| `.coverage`      | Coverage tool data file.                        |
| `.DS_Store`      | macOS file system metadata (should be ignored). |

---

### 🔹 9. **Advanced Patterns (for large projects)**

| Pattern / Structure | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| `core/settings/`    | Split settings (`base.py`, `dev.py`, `prod.py`).           |
| `apps/`             | Organize all apps inside an `apps/` folder.                |
| `config/`           | Rename `projectname/` to `config/` for clarity.            |
| `services/`         | External service clients (e.g., payment, API integration). |
| `schemas/`          | Custom schema definitions (e.g., for DRF, GraphQL).        |

---
