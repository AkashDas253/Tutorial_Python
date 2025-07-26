## Creating Models in Django 

---

### Purpose

Models in Django are Python classes used to define the structure and behavior of your database tables via Django’s ORM (Object-Relational Mapping). They act as the **single source of truth** for database schema and interaction logic.

---

### Workflow Summary

1. Define model in `models.py`
2. Register in `admin.py` (optional for admin)
3. Create migrations (`makemigrations`)
4. Apply migrations (`migrate`)
5. Use ORM to query/update data

---

### Model Class Essentials

Each model inherits from `django.db.models.Model`.

```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=200)
    published = models.DateField()
```

---

### Common Field Types

| Field Type                                | Description            |
| ----------------------------------------- | ---------------------- |
| `CharField`                               | Short strings          |
| `TextField`                               | Long text              |
| `IntegerField`                            | Integers               |
| `FloatField`                              | Floating-point numbers |
| `BooleanField`                            | True/False             |
| `DateField`, `TimeField`, `DateTimeField` | Dates and times        |
| `EmailField`, `URLField`, `SlugField`     | Formatted strings      |
| `ForeignKey`                              | One-to-many relation   |
| `ManyToManyField`                         | Many-to-many relation  |
| `OneToOneField`                           | One-to-one relation    |
| `FileField`, `ImageField`                 | File uploads           |

---

### Field Options

| Option                      | Description                 |
| --------------------------- | --------------------------- |
| `null=True`                 | Allows NULL in DB           |
| `blank=True`                | Field not required in forms |
| `default=...`               | Sets a default value        |
| `unique=True`               | Ensures uniqueness          |
| `choices=[()]`              | Enum-like behavior          |
| `verbose_name`, `help_text` | Improves admin/form UI      |

---

### Model Methods

| Method Type          | Purpose                                  |
| -------------------- | ---------------------------------------- |
| `__str__()`          | Display name in admin/logs               |
| `get_absolute_url()` | Link for instance (used in views)        |
| Custom methods       | Business logic (e.g., `.is_available()`) |

---

### Relationships

```python
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
```

---

### Meta Options

Control model behavior with nested `class Meta`.

```python
class Meta:
    ordering = ['-published']
    verbose_name = 'Book'
    db_table = 'book_table'
    unique_together = ['title', 'published']
```

---

### Best Practices

* Use singular class names (`Book`, not `Books`)
* Always define `__str__()` for readability
* Use `related_name` for ForeignKeys if needed
* Avoid hardcoded logic in models (use services)
* Index frequently queried fields (`db_index=True`)
* Keep business logic separate unless highly related

---

### Admin Registration

```python
from django.contrib import admin
from .models import Book

@admin.register(Book)
class BookAdmin(admin.ModelAdmin):
    list_display = ('title', 'published')
```

---

### Folder Context

* `app/models.py` – Model definitions
* `app/admin.py` – Optional admin interface
* `app/migrations/` – Auto-generated migration files

---

### Migration Steps Recap

```bash
python manage.py makemigrations
python manage.py migrate
```

---

### Advanced Concepts

| Concept             | Use Case                                         |
| ------------------- | ------------------------------------------------ |
| `AbstractBaseUser`  | Custom user models                               |
| `Model inheritance` | Reuse base fields in other models                |
| `Proxy models`      | Change behavior, not structure                   |
| `Signals`           | Trigger logic on save/delete                     |
| `Managers`          | Custom querysets (e.g., `Book.objects.active()`) |

---
