## Deploying Django on Heroku 

### Why Use Heroku?

* PaaS (Platform as a Service): Abstracts away infrastructure
* Easy integration with Git and GitHub
* Offers PostgreSQL, Redis, storage, add-ons, free tier (limited)
* Scalable and CI/CD-friendly

---

## Pre-requisites

| Requirement    | Description                                                                      |
| -------------- | -------------------------------------------------------------------------------- |
| Django project | A working Django project (e.g., `myproject`)                                     |
| Git            | Project must be a Git repository                                                 |
| Heroku CLI     | Install from [Heroku CLI Docs](https://devcenter.heroku.com/articles/heroku-cli) |
| Heroku Account | Free account from [heroku.com](https://heroku.com)                               |

---

## Step-by-Step Deployment Guide

---

### 1. Setup Django Project for Heroku

#### a. Install required packages

```bash
pip install gunicorn psycopg2-binary whitenoise python-decouple dj-database-url
```

#### b. `requirements.txt`

```bash
pip freeze > requirements.txt
```

#### c. `Procfile` (no extension)

```
web: gunicorn myproject.wsgi
```

#### d. `runtime.txt` (optional but recommended)

```
python-3.11.3
```

---

### 2. Configure `settings.py`

#### a. Static file settings

```python
import dj_database_url
from decouple import config

STATIC_ROOT = BASE_DIR / 'staticfiles'
STATIC_URL = '/static/'

# WhiteNoise Middleware
MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'
```

#### b. Security and Debug settings

```python
DEBUG = config('DEBUG', default=False, cast=bool)
SECRET_KEY = config('SECRET_KEY')
ALLOWED_HOSTS = ['.herokuapp.com', 'your_custom_domain.com']
```

#### c. Database config (override default)

```python
DATABASES['default'] = dj_database_url.config(conn_max_age=600, ssl_require=True)
```

---

### 3. Environment Variables

Create a `.env` file locally (don't commit it):

```
DEBUG=False
SECRET_KEY=your_production_secret
```

Use `python-decouple` to load it.

On Heroku, set using:

```bash
heroku config:set SECRET_KEY=your_production_secret
```

---

### 4. Initialize Git Repo (if not done)

```bash
git init
git add .
git commit -m "Initial commit"
```

---

### 5. Login & Create Heroku App

```bash
heroku login
heroku create your-app-name
```

---

### 6. Push to Heroku

```bash
git push heroku master  # or 'main' if you're on main branch
```

---

### 7. Setup Heroku PostgreSQL

```bash
heroku addons:create heroku-postgresql:hobby-dev
```

Heroku sets the `DATABASE_URL` env variable.

---

### 8. Apply Migrations

```bash
heroku run python manage.py migrate
```

---

### 9. Collect Static Files

```bash
heroku run python manage.py collectstatic --noinput
```

---

### 10. Create Superuser (Optional)

```bash
heroku run python manage.py createsuperuser
```

---

### 11. Open the App

```bash
heroku open
```

---

## Suggested Project Structure for Heroku

```
myproject/
│
├── myproject/               # Django project
│   ├── __init__.py
│   ├── settings.py          # Use decouple + dj-database-url here
│   ├── urls.py
│   └── wsgi.py
│
├── manage.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── .env                     # Not committed
```

---

### Optional: Custom Domain + SSL (Free)

```bash
heroku domains:add www.yourdomain.com
# Use Cloudflare or Namecheap to point DNS
```

---

### Optional: Debugging

```bash
heroku logs --tail
```

---

### Optional: Add-ons

| Feature    | Add-on             |
| ---------- | ------------------ |
| DB Backup  | Heroku PG Backups  |
| Caching    | Heroku Redis       |
| Monitoring | LogDNA, Papertrail |
| Email      | SendGrid, Mailgun  |

---
