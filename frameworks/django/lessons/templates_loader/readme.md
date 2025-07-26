## Django Template Loader

---

### Purpose

The **template loader** in Django is responsible for **searching, locating, and loading** template files (like `.html`) from configured locations based on the template name passed to functions like `render()` or `loader.get_template()`.

---

### How Template Loading Works

1. The view calls `render()` or `loader.get_template(template_name)`
2. Django uses a list of **loaders** to search for the template in order.
3. First match is returned and rendered.

---

### Template Discovery Sources

Django looks for templates in:

* `DIRS` in `TEMPLATES` setting
* `APP_DIRS` templates (i.e., `<app>/templates/`)
* Custom loaders (if provided)

---

### `TEMPLATES` Setting in `settings.py`

```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / "templates"],  # Global template dirs
        'APP_DIRS': True,  # Enables app-level loading
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                # ...
            ],
        },
    },
]
```

---

### Loaders – Built-in Options

| Loader Name                                      | Description                             |
| ------------------------------------------------ | --------------------------------------- |
| `django.template.loaders.filesystem.Loader`      | Loads templates from `DIRS`             |
| `django.template.loaders.app_directories.Loader` | Loads from `<app>/templates/`           |
| `django.template.loaders.cached.Loader`          | Caches templates to improve performance |

---

### Template Directory Structure (Best Practice)

```
project/
├── app1/
│   └── templates/
│       └── app1/
│           └── template1.html
├── templates/         <-- global templates
│   └── base.html
```

---

### Using Loader Manually (Alternative to `render()`)

```python
from django.template import loader
from django.http import HttpResponse

def my_view(request):
    template = loader.get_template('myapp/hello.html')
    return HttpResponse(template.render({'name': 'Subham'}, request))
```

---

### Cached Loader Example

```python
'OPTIONS': {
    'loaders': [
        ('django.template.loaders.cached.Loader', [
            'django.template.loaders.filesystem.Loader',
            'django.template.loaders.app_directories.Loader',
        ]),
    ],
}
```

---

### Loader Flow – Mermaid Diagram

```mermaid
flowchart TD
    A[render('template.html')] --> B[Template Loader]
    B --> C[Check DIRS]
    B --> D[Check APP_DIRS]
    B --> E[Use Custom Loaders (if any)]
    C -->|Found| F[Return Template]
    D -->|Found| F
    E -->|Found| F
    F --> G[Render with Context]
```

---

### Error Handling

* If no template is found: `TemplateDoesNotExist` exception.
* If template is invalid: `TemplateSyntaxError`.

---
