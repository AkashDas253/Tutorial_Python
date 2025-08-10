## **Serving Static Files in Django**

### **1. Enable `django.contrib.staticfiles`**

Make sure it's in your `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',  # Required for static files
]
```

---

### **2. Define Static Files Settings in `settings.py`**

```python
# URL prefix for static files
STATIC_URL = '/static/'

# Directory for collected static files (used in production)
STATIC_ROOT = BASE_DIR / 'staticfiles'

# Extra directories to search for static files
STATICFILES_DIRS = [
    BASE_DIR / 'static',
]
```

---

### **3. Place Your Static Files**

* Create a `static/` folder in your app or project root.
* Example:

  ```
  project/
    static/
      css/
        style.css
      js/
        script.js
  ```

---

### **4. Reference Static Files in Templates**

Load static template tag and reference the file:

```django
{% load static %}
<link rel="stylesheet" href="{% static 'css/style.css' %}">
<script src="{% static 'js/script.js' %}"></script>
```

---

### **5. Development Mode (Automatic Serving)**

* Django automatically serves static files in development when `DEBUG = True`.
* Run:

```bash
python manage.py runserver
```

---

### **6. Production Mode**

* Run `collectstatic` to gather all static files in `STATIC_ROOT`:

```bash
python manage.py collectstatic
```

* Serve them with:

  * A web server (e.g., Nginx, Apache)
  * Middleware like **WhiteNoise**
  * A CDN

---
