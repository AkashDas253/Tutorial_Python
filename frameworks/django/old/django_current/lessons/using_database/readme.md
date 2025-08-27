## Using a Database in Django

---

### Purpose

Django uses a database to store and manage:

* App-specific data via models
* User authentication and session data
* Admin panel content
* Schema and version tracking via migrations
* API data, logs, relationships, and third-party integrations

---

### Core Concepts

| Concept                        | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| ORM (Object Relational Mapper) | Interacts with the database using Python classes and methods      |
| Models                         | Python classes that define database tables (schema)               |
| Migrations                     | Versioned schema files to reflect changes to models in the DB     |
| QuerySets                      | Lazy evaluation queries to retrieve/filter/update/delete records  |
| Admin Integration              | Automatic admin UI for DB records via model registration          |
| Related Data                   | ForeignKey, OneToOneField, ManyToManyField for data relationships |

---

### Database Usage Lifecycle

#### Step 1: Define Model

```python
# models.py
class Book(models.Model):
    title = models.CharField(max_length=200)
    published = models.DateField()
```

#### Step 2: Run Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

This generates migration files and syncs them with your database.

#### Step 3: CRUD Operations via ORM

```python
# Create
Book.objects.create(title="Django Mastery", published="2025-07-01")

# Read
Book.objects.get(id=1)

# Update
book = Book.objects.get(id=1)
book.title = "Updated Title"
book.save()

# Delete
book.delete()
```

#### Step 4: Use in Views or APIs

```python
def all_books(request):
    books = Book.objects.all()
    return render(request, 'books.html', {'books': books})
```

---

### Admin Integration

```python
# admin.py
from .models import Book
admin.site.register(Book)
```

---

### Working with Related Data

```python
class Author(models.Model):
    name = models.CharField(max_length=100)

class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.ForeignKey(Author, on_delete=models.CASCADE)
```

---

### Advanced Usage

| Feature       | Example / Use Case                               |                      |
| ------------- | ------------------------------------------------ | -------------------- |
| Raw SQL       | `Book.objects.raw("SELECT * FROM app_book")`     |                      |
| Aggregation   | `Book.objects.aggregate(Avg("pages"))`           |                      |
| F Expressions | `Book.objects.filter(price__gt=F("cost"))`       |                      |
| Q Objects     | \`Book.objects.filter(Q(pages\_\_gt=100)         | Q(pages\_\_lt=20))\` |
| Transactions  | `with transaction.atomic(): ...`                 |                      |
| Indexing      | `Meta: indexes = [...]` for DB-level performance |                      |

---

### Tools and Management

| Command                  | Use                                      |
| ------------------------ | ---------------------------------------- |
| `dbshell`                | Open interactive DB shell                |
| `inspectdb`              | Reverse engineer existing DB into models |
| `flush`                  | Deletes all data and resets sequences    |
| `showmigrations`         | Shows all migrations                     |
| `sqlmigrate <app> <id>`  | Shows SQL for a specific migration       |
| `migrate --plan`         | Preview migration plan                   |
| `makemigrations --merge` | Resolve conflicting migrations           |

---

### Multi-Database Support (Optional)

```python
# settings.py
DATABASES = {
    'default': {...},
    'analytics': {...}
}
```

Use it in ORM:

```python
Model.objects.using('analytics').all()
```

---

### Supported Backends

| DB Backend    | Support Status | Notes                         |
| ------------- | -------------- | ----------------------------- |
| SQLite        | ✅ Default      | Good for dev/testing          |
| PostgreSQL    | ✅ Recommended  | Full-featured, production use |
| MySQL/MariaDB | ✅ Supported    | Needs additional tuning       |
| Oracle        | ✅ Supported    | Enterprise, less common       |
| Others        | ⚠️ Indirect    | Via custom backends/adapters  |

---
