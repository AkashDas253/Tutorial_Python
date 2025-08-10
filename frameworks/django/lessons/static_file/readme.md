## Static Files in Django

### Overview

Static files are assets like CSS, JavaScript, and images that don’t change dynamically and are served to clients as-is. Django provides a structured way to manage and serve them in development and production.

---

### Key Concepts

* **Static Files**: Non-Python files used for UI, styling, and client-side interactivity.
* **`STATIC_URL`**: Base URL for serving static files.
* **`STATICFILES_DIRS`**: Additional directories to look for static files apart from each app’s `static` directory.
* **`STATIC_ROOT`**: The single directory where all static files are collected during deployment using `collectstatic`.
* **`django.contrib.staticfiles`**: App responsible for managing static files.
* **App-level Static Directory**: Each app can have its own `static` folder for organizing its files.

---

### Static Files in Development

* Static files are served automatically when `DEBUG=True` and `django.contrib.staticfiles` is in `INSTALLED_APPS`.
* Located in `app_name/static/app_name/` or global static directories.

---

### Static Files in Production

* Collected into `STATIC_ROOT` using:

```bash
python manage.py collectstatic
```

* Served by a web server (e.g., Nginx, Apache) instead of Django for performance.

---

### Usage in Templates

```django
{% load static %}  <!-- Load static template tag -->
<link rel="stylesheet" href="{% static 'css/styles.css' %}">
<img src="{% static 'images/logo.png' %}" alt="Logo">
```

---

### Settings Example (`settings.py`)

```python
STATIC_URL = '/static/'  # URL prefix for static files
STATICFILES_DIRS = [BASE_DIR / 'static']  # Additional static file dirs
STATIC_ROOT = BASE_DIR / 'staticfiles'    # Where collectstatic will store files
```

---

### Commands for Static File Management

```bash
python manage.py findstatic file_name   # Locate static file path
python manage.py collectstatic          # Gather all static files into STATIC_ROOT
```

---
