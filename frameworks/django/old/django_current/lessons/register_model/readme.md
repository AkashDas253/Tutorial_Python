## Registering a Model to Django Admin 

### Purpose

Registering a model with the **Django Admin site** allows you to **view, add, edit, and delete** that model’s data through Django’s built-in admin interface.

---

### Requirements

* `django.contrib.admin` must be in `INSTALLED_APPS`
* `django.contrib.auth` and `django.contrib.contenttypes` must also be installed (for permissions system)
* Admin migrations applied:

  ```bash
  python manage.py migrate
  ```
* Superuser account created to access admin

---

### Steps to Register a Model

| Step  | Action                                 | What Happens Internally                                 |
| ----- | -------------------------------------- | ------------------------------------------------------- |
| **1** | Import the model in `admin.py`         | Makes it available for registration                     |
| **2** | Use `admin.site.register(Model)`       | Registers model class with the admin site               |
| **3** | (Optional) Customize with `ModelAdmin` | Defines how the model is displayed and managed in admin |
| **4** | Access admin at `/admin/`              | Model is listed and manageable                          |

---

### Basic Registration

```python
# admin.py
from django.contrib import admin
from .models import Book

admin.site.register(Book)
```

---

### Customizing Display with `ModelAdmin`

```python
# admin.py
from django.contrib import admin
from .models import Book

class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'author', 'published_date')  # Columns in list view
    search_fields = ('title', 'author')                   # Search box fields
    list_filter = ('published_date',)                     # Sidebar filters
    ordering = ('-published_date',)                       # Default ordering

admin.site.register(Book, BookAdmin)
```

---

### Inline Model Registration

* Used to edit related models directly within a parent model form in admin.

```python
from django.contrib import admin
from .models import Book, Review

class ReviewInline(admin.TabularInline):
    model = Review
    extra = 1

class BookAdmin(admin.ModelAdmin):
    inlines = [ReviewInline]

admin.site.register(Book, BookAdmin)
```

---

### How Registration Works Internally

1. When Django starts, it loads all apps listed in `INSTALLED_APPS`.
2. If an app contains `admin.py`, it executes it.
3. `admin.site.register()` adds the model to the `AdminSite` registry.
4. The model appears in the `/admin/` interface for any **user with permissions** (superusers see all).

---

### Best Practices

* Always create a **custom `ModelAdmin` class** for better display and usability.
* Use `search_fields`, `list_filter`, and `list_display` for large datasets.
* Limit fields editable in admin with `readonly_fields` and `exclude`.
* Use `inlines` for related model editing.
* Avoid registering sensitive models unless necessary.

---
