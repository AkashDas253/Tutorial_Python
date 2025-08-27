## Deployment on PythonAnywhere 

### Prerequisites

* Django project created and working locally
* A PythonAnywhere account
* GitHub repo of your project (optional but recommended)
* Virtual environment with required packages installed (`pip freeze > requirements.txt`)

---

### Step-by-Step Deployment

#### 1. **Upload or Clone Your Project**

* **Via GitHub (recommended)**:

  * Go to **Bash console** in PythonAnywhere
  * Clone your project:

    ```bash
    git clone https://github.com/username/your-django-project.git
    ```
* **Via manual upload**:

  * Use the **Files** tab to upload your zipped project, then extract it in the PythonAnywhere file manager.

#### 2. **Set Up Virtual Environment**

```bash
cd your-django-project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 3. **Configure WSGI**

* Go to **Web tab > WSGI configuration file** (linked in the Web tab)
* Modify the `sys.path` and set up your Django app:

```python
import sys
path = '/home/yourusername/your-django-project'
if path not in sys.path:
    sys.path.append(path)

from your_django_project.wsgi import application
```

#### 4. **Configure Web App**

* Go to **Web tab** > Add a new Web App
* Choose:

  * **Manual configuration**
  * Python version matching your project
* Set:

  * **Source code location**: `/home/yourusername/your-django-project`
  * **Virtualenv**: `/home/yourusername/your-django-project/venv`

#### 5. **Update Static and Media Settings**

In `settings.py`:

```python
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
```

Then run:

```bash
python manage.py collectstatic
```

In the **Web tab**:

* Add static and media file mappings:

  * `/static/` → `/home/yourusername/your-django-project/static/`
  * `/media/` → `/home/yourusername/your-django-project/media/`

#### 6. **Set Environment Variables**

In the **Web tab > Environment Variables**, add:

* `DJANGO_SETTINGS_MODULE=your_django_project.settings`
* Any other secrets (e.g., `SECRET_KEY`, `DEBUG=False`, etc.)

#### 7. **Database Migration**

In Bash:

```bash
python manage.py migrate
```

#### 8. **Reload Web App**

* Press **Reload** button on the Web tab to apply all changes.

---

### Notes

* Use `.env` for secrets; load them using `python-decouple` or `os.environ`.
* Set `ALLOWED_HOSTS = ['yourusername.pythonanywhere.com']`
* Use `DEBUG = False` in production.

---
