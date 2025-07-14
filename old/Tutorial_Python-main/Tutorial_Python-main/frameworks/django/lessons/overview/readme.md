## Overview of Django

### Definition

**Django** is a high-level Python web framework that enables rapid development of secure and maintainable websites. It follows the **Model-View-Template (MVT)** architectural pattern and emphasizes **reusability**, **pluggability**, and the **DRY (Don’t Repeat Yourself)** principle.

---

### Key Features

* **MVT Architecture**: Separates data (Model), user interface (Template), and logic (View).
* **ORM (Object-Relational Mapping)**: Maps Python classes to database tables.
* **Admin Interface**: Auto-generated web-based admin panel.
* **Security**: Built-in protection against CSRF, XSS, SQL Injection, clickjacking.
* **Scalability**: Supports scalable applications with caching, load balancing, etc.
* **Built-in Development Server**: For local development and testing.
* **Internationalization and Localization**: Multilingual support.
* **Middleware Support**: Hooks into request/response processing.
* **Form Handling**: HTML form generation and validation.

---

### Architecture (MVT Pattern)

| Layer        | Description                               |
| ------------ | ----------------------------------------- |
| **Model**    | Defines data structure and business logic |
| **View**     | Processes requests and returns responses  |
| **Template** | Handles the presentation layer (HTML)     |

---

### Core Components

* **Models**: Python classes representing database tables
* **Views**: Functions or classes that handle HTTP requests
* **Templates**: HTML files with Django Template Language (DTL)
* **URLs**: Routing configuration using `urlpatterns`
* **Forms**: Built-in support for form creation and validation
* **Admin**: Configurable interface for managing data

---

### Project Structure (Typical)

```
project/
├── manage.py
├── project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── app/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   └── static/
```

---

### Supported Databases

* PostgreSQL
* MySQL
* SQLite (default)
* Oracle

---

### Built-in Commands (`manage.py`)

* `runserver`: Start development server
* `makemigrations`: Create migration files
* `migrate`: Apply migrations
* `createsuperuser`: Create admin user
* `startapp`: Create a new app
* `collectstatic`: Gather static files

---

### Template Language (DTL)

* Control structures: `{% if %}`, `{% for %}`
* Template inheritance: `{% extends %}`, `{% block %}`
* Filters: `{{ value|filter }}` (e.g., `{{ name|lower }}`)

---

### Middleware Examples

* `SecurityMiddleware`
* `SessionMiddleware`
* `AuthenticationMiddleware`
* `CommonMiddleware`
* Custom middleware

---

### Security Features

* CSRF protection
* SQL injection prevention via ORM
* XSS prevention via template auto-escaping
* Clickjacking protection via headers
* Secure password storage using PBKDF2

---

### REST and API Support

* Use **Django REST Framework (DRF)** for building REST APIs

  * Serializers
  * ViewSets
  * Routers
  * Authentication (Token, Session, JWT)

---

### Deployment Tools and Considerations

* Use **Gunicorn**, **uWSGI** with **Nginx**
* Configure `ALLOWED_HOSTS`, `DEBUG`, `SECURE_*` settings
* Use **WhiteNoise** or S3 for static/media files
* Database configuration with environment variables

---

### Common Use Cases

* Content Management Systems (CMS)
* Social media platforms
* eCommerce websites
* SaaS applications
* RESTful APIs and microservices

---

### Popular Extensions and Tools

* **Django REST Framework (DRF)**: API development
* **Celery**: Task queues
* **Django Channels**: WebSocket support
* **Django Allauth**: Authentication (OAuth, etc.)
* **Whitenoise**: Serve static files
* **Django Debug Toolbar**: Debugging and performance

---
