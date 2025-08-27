## WhiteNoise in Django

### Purpose

* Serves static files directly from your Django application in production without relying on an external web server like Nginx.
* Simplifies deployment, especially for platforms like Heroku.

---

## Installation

```bash
pip install whitenoise
```

---

## Setup in Django

**In `settings.py`:**

```python
# Add whitenoise middleware *after* SecurityMiddleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    # ... other middleware
]

# Enable compressed static files
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Static files settings
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
```

---

## Usage Flow

* **Collect static files**

```bash
python manage.py collectstatic
```

* WhiteNoise serves files from `STATIC_ROOT` automatically.
* Handles caching and gzip/brotli compression.

---

## Key Features

* **Compression**: Gzip and Brotli support for smaller file size.
* **Immutable file names**: Uses hashed filenames for cache-busting.
* **No extra config**: Works directly via middleware.
* **Better performance**: Files served with proper caching headers.

---

## Deployment Notes

* Ideal for small to medium apps, or when using PaaS (Heroku, Render).
* For large-scale/high-traffic apps, use it behind a dedicated static file server/CDN.

---

## Example Project Structure

```
myproject/
│── manage.py
│── myproject/
│   │── settings.py
│   │── urls.py
│   │── wsgi.py
│
│── static/
│   │── css/
│   │── js/
│
│── staticfiles/  # Created after collectstatic
```

---
