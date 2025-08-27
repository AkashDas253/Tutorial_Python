## Django Messaging Framework – Setup and Usage

---

### Setup

#### 1. Install Required App

```python
# settings.py
INSTALLED_APPS = [
    'django.contrib.messages',  # Enables messaging framework
]
```

#### 2. Add Middleware

```python
MIDDLEWARE = [
    'django.contrib.sessions.middleware.SessionMiddleware',  # Required if using SessionStorage
    'django.contrib.messages.middleware.MessageMiddleware',  # Enables request/response messaging
]
```

#### 3. Configure Message Storage Backend

```python
# Default: uses cookies first, falls back to sessions
MESSAGE_STORAGE = 'django.contrib.messages.storage.fallback.FallbackStorage'

# Options:
# 'django.contrib.messages.storage.cookie.CookieStorage'
# 'django.contrib.messages.storage.session.SessionStorage'
```

#### 4. Enable Template Context Processor

```python
# settings.py
TEMPLATES = [
    {
        'OPTIONS': {
            'context_processors': [
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```

This makes `messages` available in templates.

---

### Usage

#### Adding Messages in Views

```python
from django.contrib import messages

messages.debug(request, "Debugging details")     # Debug level
messages.info(request, "General information")    # Info level
messages.success(request, "Operation successful") # Success level
messages.warning(request, "This is a warning")   # Warning level
messages.error(request, "An error occurred")     # Error level
```

#### Retrieving Messages in Templates

```html
{% if messages %}
  <ul>
    {% for message in messages %}
      <li class="{{ message.tags }}">{{ message }}</li>
    {% endfor %}
  </ul>
{% endif %}
```

* `message` → Text of the message.
* `message.tags` → CSS class based on level (`success`, `error`, etc.).

---

### Customization

#### Custom Tags / Levels

```python
from django.contrib.messages import add_message, INFO

add_message(request, INFO, "Custom info message")
```

#### Styling with CSS Frameworks

* Bootstrap → `alert alert-{{ message.tags }}`
* Tailwind → `bg-{{ message.tags }}-500 text-white p-2`

---

### Integration Examples

* **Authentication** → `messages.success(request, "Logged in successfully")`
* **Forms** → On form validation failure → `messages.error(request, "Please correct errors")`
* **CRUD Operations** → After saving/deleting records → `messages.success(request, "Record deleted")`

---

✅ **Setup ensures middleware, storage, and template integration are configured.**
✅ **Usage allows adding messages in views and displaying them in templates.**

---
