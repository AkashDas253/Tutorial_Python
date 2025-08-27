## Django Templates – Comprehensive Overview

---

### What is a Template?

A **template** in Django is a text file (usually HTML) that defines the **structure and layout** of a web page and uses Django Template Language (DTL) to dynamically insert data and control content flow.

---

### Key Components

* **Template File:** HTML + DTL syntax (`.html`)
* **Context:** A dictionary of variables passed to the template
* **Template Engine:** Renders the template using the context
* **Loader:** Finds and loads templates from configured locations

---

### Workflow of Using Templates

1. **Create template file** (e.g., `home.html`)
2. **Define view function** and pass context
3. **Use `render()`** to combine template and context
4. **Return `HttpResponse`** with rendered content

---

### Basic Syntax in Template

| Type     | Syntax           | Purpose                                |
| -------- | ---------------- | -------------------------------------- |
| Variable | `{{ variable }}` | Inserts variable value                 |
| Tag      | `{% tag %}`      | Executes logic (loop, condition, etc.) |
| Comment  | `{# comment #}`  | Adds comments, not rendered            |

---

### Common Template Tags

| Tag                         | Purpose                         |
| --------------------------- | ------------------------------- |
| `{% if condition %}`        | Conditional rendering           |
| `{% for item in list %}`    | Loop over items                 |
| `{% include 'file.html' %}` | Include external template       |
| `{% extends 'base.html' %}` | Inherit from parent             |
| `{% block name %}`          | Define replaceable section      |
| `{% load static %}`         | Load static file support        |
| `{% csrf_token %}`          | Protect forms from CSRF attacks |
| `{% url 'route_name' %}`    | URL reversing                   |

---

### Common Filters

| Filter Example                      | Purpose               |
| ------------------------------------ | --------------------- |
| `{{ name  | lower }}`                | Convert to lowercase  |
| `{{ list  | length }}`               | Get length            |
| `{{ date  | date:"Y-m-d" }}`         | Format date           |
| `{{ value | default:"N/A" }}`        | Default if empty      |

---

### Template Inheritance

#### `base.html`

```html
<html>
<head>
  <title>{% block title %}MySite{% endblock %}</title>
</head>
<body>
  {% block content %}{% endblock %}
</body>
</html>
```

#### `child.html`

```html
{% extends 'base.html' %}
{% block title %}Homepage{% endblock %}
{% block content %}
  <h1>Welcome, {{ user }}</h1>
{% endblock %}
```

---

### Static File Usage

```html
{% load static %}
<link rel="stylesheet" href="{% static 'css/style.css' %}">
<img src="{% static 'img/logo.png' %}">
```

> Requires `STATICFILES_DIRS` and `STATIC_URL` to be set in `settings.py`

---

### Directory Structure

Best Practice:

```
project/
├── app/
│   └── templates/
│       └── app/
│           └── page.html
├── templates/
│   └── base.html
```

> Django finds templates via `DIRS` and `APP_DIRS` options in `TEMPLATES` setting.

---

### Using Templates in Views

```python
from django.shortcuts import render

def home(request):
    return render(request, 'home.html', {'user': 'Subham'})
```

---

### Manual Rendering (Optional)

```python
from django.template import loader
template = loader.get_template('home.html')
html = template.render({'name': 'Subham'})
```

---

### Custom Template Tags and Filters

Create `templatetags` inside your app:

```bash
app/
├── templatetags/
│   ├── __init__.py
│   └── custom_tags.py
```

```python
# custom_tags.py
from django import template
register = template.Library()

@register.filter
def double(value):
    return value * 2
```

Use in template:

```html
{% load custom_tags %}
{{ 5|double }}
```

---

### Context Processors (Auto-available Data)

Add global variables:

```python
# settings.py
TEMPLATES['OPTIONS']['context_processors'] += [
    'django.template.context_processors.request',
]
```

---

### Error Handling

| Error                    | Cause                                         |
| ------------------------ | --------------------------------------------- |
| `TemplateDoesNotExist`   | Wrong path or not found                       |
| `TemplateSyntaxError`    | Invalid tag or filter                         |
| `ContextMissingVariable` | Missing variable in context (safe by default) |

---

### Advantages of Django Templates

* Clean separation of logic and presentation
* Easy inheritance and reuse
* Powerful control with tags and filters
* Secure by default (auto escaping)

---
