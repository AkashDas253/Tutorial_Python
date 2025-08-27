## Deployment of Django Applications 

### Goals of Deployment

* Make the application accessible over the internet
* Ensure performance, scalability, and security
* Setup services to serve static/media files
* Configure WSGI servers and web servers

---

### Preparation Before Deployment

* Set `DEBUG = False` in `settings.py`
* Add production `ALLOWED_HOSTS`
* Use `env` variables (e.g., via `python-decouple`, `dotenv`)
* Configure `DATABASES` to use production DB (e.g., PostgreSQL)
* Set up static and media file handling
* Secure `SECRET_KEY`, DB credentials

---

### Deployment Environment Options

| Platform Type     | Tools/Services                  |
| ----------------- | ------------------------------- |
| IaaS (custom VPS) | Gunicorn/uWSGI + Nginx          |
| PaaS              | Heroku, PythonAnywhere, Render  |
| Containerized     | Docker + Gunicorn/uWSGI + Nginx |
| Serverless        | Zappa (for AWS Lambda)          |

---

### Web Server Gateway Interface (WSGI)

* WSGI allows web servers to communicate with Django
* Two common WSGI servers:

  * **Gunicorn** (Production-ready, easy to use)
  * **uWSGI** (More configurable, widely adopted)

---

### Typical VPS Deployment (Gunicorn + Nginx)

**Steps:**

* SSH into server
* Clone project from repo
* Set up virtual environment & install dependencies
* Configure `.env`
* Apply DB migrations
* Collect static files using `python manage.py collectstatic`
* Test with Gunicorn
* Configure Nginx as a reverse proxy
* Set up systemd service for Gunicorn
* Configure firewall (e.g., `ufw allow 'Nginx Full'`)
* Enable HTTPS via Let’s Encrypt (`certbot`)

---

### Static and Media Files in Production

* `STATIC_ROOT` must be defined
* Use `python manage.py collectstatic`
* Serve:

  * Static: via Nginx or WhiteNoise
  * Media: via Nginx or cloud storage (e.g., S3)

---

### Using Docker

* Dockerfile defines app environment
* `docker-compose.yml` for multiple services (DB, Redis, etc.)
* Use `gunicorn` in container
* Optional: Nginx as reverse proxy container

---

### Deployment via Heroku (Example PaaS)

* Install Heroku CLI
* Add `Procfile` → `web: gunicorn myproject.wsgi`
* Add `runtime.txt` → specify Python version
* Add `requirements.txt` & `collectstatic` hook
* Set env variables via Heroku dashboard/CLI
* Use `heroku config:set DEBUG=False`
* Use Heroku PostgreSQL, Redis add-ons if needed

---

### Optional Deployment Add-ons

| Feature    | Tool/Service                        |
| ---------- | ----------------------------------- |
| HTTPS      | Let’s Encrypt (certbot), Heroku SSL |
| CI/CD      | GitHub Actions, GitLab CI/CD        |
| Monitoring | Sentry, Prometheus, New Relic       |
| Logging    | ELK stack, Papertrail               |
| Caching    | Redis, Memcached                    |

---

### Common Post-Deployment Checklist

* [ ] DEBUG = False
* [ ] Strong SECRET\_KEY
* [ ] ALLOWED\_HOSTS configured
* [ ] Proper HTTPS enabled
* [ ] Database secured and backed up
* [ ] Logs monitored
* [ ] Performance optimized

---
