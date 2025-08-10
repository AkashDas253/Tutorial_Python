## **Serving Static Files with WhiteNoise**

### **Purpose**

WhiteNoise allows Django to serve its own static files efficiently, without needing a separate web server like Nginx in production.

---

## **Installation**

```bash
pip install whitenoise
```

---

## **Configuration**

### **Add to Installed Apps**

```python
INSTALLED_APPS = [
    'django.contrib.staticfiles',  # Required for static file handling
    # other apps...
]
```

---

### **Update Middleware**

WhiteNoise must be placed **right after `SecurityMiddleware`** in the middleware list:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # WhiteNoise middleware
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # other middlewares...
]
```

---

### **Static File Settings**

```python
STATIC_URL = '/static/'  # URL path for static files
STATIC_ROOT = BASE_DIR / 'staticfiles'  # Location where collectstatic stores files
STATICFILES_DIRS = [
    BASE_DIR / 'static',  # Local static files for development
]
```

---

### **Enable Gzip and Brotli Compression (Optional but Recommended)**

```python
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
```

* **Gzip**: Reduces file size for faster delivery.
* **Manifest**: Adds unique hashes to filenames for cache-busting.

---

## **Collect Static Files**

Before running the server in production:

```bash
python manage.py collectstatic
```

This will copy all static files from apps and `STATICFILES_DIRS` into `STATIC_ROOT`.

---

## **Run Server**

WhiteNoise works with:

```bash
python manage.py runserver
```

In production, it serves files directly through WSGI (e.g., Gunicorn).

---

## **Usage in Templates**

```html
{% load static %}
<link rel="stylesheet" href="{% static 'css/style.css' %}">
<img src="{% static 'images/logo.png' %}" alt="Logo">
```

---

## **Advantages of WhiteNoise**

* No need for a separate static file server in production.
* Gzip/Brotli compression support.
* Cache-busting via Manifest storage.
* Simple setup.

---
