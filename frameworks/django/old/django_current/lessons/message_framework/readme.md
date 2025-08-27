## **Messages Framework in Django**

The **messages framework** lets you store temporary messages (success, error, info, warning, debug) for the user, which are cleared after being displayed once.

---

### **Enabling the Messages Framework**

* **Add to `INSTALLED_APPS`** (default in most projects):

  ```python
  INSTALLED_APPS = [
      'django.contrib.messages',
  ]
  ```
* **Add Middleware** in `MIDDLEWARE`:

  ```python
  MIDDLEWARE = [
      'django.contrib.sessions.middleware.SessionMiddleware',
      'django.contrib.messages.middleware.MessageMiddleware',
  ]
  ```
* **Context Processor** (in `TEMPLATES`):

  ```python
  'django.contrib.messages.context_processors.messages'
  ```

---

### **Message Levels**

| Level Name | Constant           | Default Tag |
| ---------- | ------------------ | ----------- |
| Debug      | `messages.DEBUG`   | `debug`     |
| Info       | `messages.INFO`    | `info`      |
| Success    | `messages.SUCCESS` | `success`   |
| Warning    | `messages.WARNING` | `warning`   |
| Error      | `messages.ERROR`   | `error`     |

---

### **Using Messages in Views**

```python
from django.contrib import messages

# Adding messages
messages.debug(request, "Debug message")       # Low-level info
messages.info(request, "Info message")         # General info
messages.success(request, "Action successful") # Success alert
messages.warning(request, "Warning alert")     # Caution
messages.error(request, "Error occurred")      # Error alert
```

---

### **Rendering Messages in Template**

```html
{% if messages %}
    <ul>
        {% for message in messages %}
            <li{% if message.tags %} class="{{ message.tags }}"{% endif %}>
                {{ message }}
            </li>
        {% endfor %}
    </ul>
{% endif %}
```

* **`message.tags`** → returns the CSS-friendly class name for styling.
* Can be styled with Bootstrap:

  ```html
  <div class="alert alert-{{ message.tags }}">{{ message }}</div>
  ```

---

### **Customizing Message Tags**

```python
from django.contrib import messages

MESSAGE_TAGS = {
    messages.ERROR: 'danger',  # Bootstrap uses 'danger' for errors
}
```

---

### **Storage Backends**

| Backend Name               | Behavior                     |
| -------------------------- | ---------------------------- |
| `SessionStorage` (default) | Stores in session            |
| `CookieStorage`            | Stores in cookies            |
| `FallbackStorage`          | Tries cookies, then sessions |

Set in `settings.py`:

```python
MESSAGE_STORAGE = 'django.contrib.messages.storage.session.SessionStorage'
```

---

### **Important Notes**

* Messages are **consumed once** — they disappear after template render.
* You can use them in **FBV** or **CBV**.
* Often used after **redirects** to pass user feedback.
* Works well with **Bootstrap Alerts** for styling.

---
