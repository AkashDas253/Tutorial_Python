## Django Models 

---

### Purpose

Django models define the structure of your database tables in Python. Each model maps to a single table.

---

### Steps to Set Up and Use Models

#### 1. **Define a Model**

In `models.py` of an app:

```python
from django.db import models

class Book(models.Model):
    title = models.CharField(max_length=255)
    author = models.CharField(max_length=100)
    published_date = models.DateField()
    is_available = models.BooleanField(default=True)
```

---

#### 2. **Activate the App and Model**

* Add your app to `INSTALLED_APPS` in `settings.py`

```python
INSTALLED_APPS = [
    ...
    'myapp',
]
```

---

#### 3. **Create Migrations**

Generate SQL based on model changes:

```bash
python manage.py makemigrations
```

---

#### 4. **Apply Migrations**

Create/update database tables:

```bash
python manage.py migrate
```

---

#### 5. **Use Models in Views / Shell**

* **In views.py**:

```python
from .models import Book

def get_books(request):
    books = Book.objects.all()
    ...
```

* **Using Django Shell**:

```bash
python manage.py shell
```

```python
from myapp.models import Book
Book.objects.create(title="Django Guide", author="John", published_date="2023-01-01")
books = Book.objects.all()
```

---

### Common Field Types

| Field Type        | Description                      |
| ----------------- | -------------------------------- |
| `CharField`       | String (requires `max_length`)   |
| `TextField`       | Large text                       |
| `IntegerField`    | Integer                          |
| `BooleanField`    | True/False                       |
| `DateField`       | Date only                        |
| `DateTimeField`   | Date and time                    |
| `EmailField`      | Validated email                  |
| `ForeignKey`      | Many-to-one relationship         |
| `ManyToManyField` | Many-to-many relationship        |
| `OneToOneField`   | One-to-one relationship          |
| `ImageField`      | Image upload (requires `Pillow`) |

---

### Model Meta Options

Inside a model:

```python
class Book(models.Model):
    ...
    class Meta:
        ordering = ['title']
        verbose_name = 'Book Entry'
```

| Meta Option           | Purpose                              |
| --------------------- | ------------------------------------ |
| `ordering`            | Default ordering                     |
| `verbose_name`        | Singular name in admin               |
| `verbose_name_plural` | Plural name in admin                 |
| `db_table`            | Custom database table name           |
| `unique_together`     | Multiple field uniqueness constraint |

---

### Useful Model Methods

```python
def __str__(self):
    return self.title
```

```python
def get_absolute_url(self):
    return reverse('book-detail', kwargs={'pk': self.pk})
```

---

### QuerySet API (Common Uses)

```python
Book.objects.all()                     # Get all books
Book.objects.filter(author="John")     # Filter
Book.objects.get(id=1)                 # Get one (raises error if not found)
Book.objects.exclude(is_available=False)
Book.objects.order_by('-published_date')
Book.objects.count()
Book.objects.exists()
```

---
