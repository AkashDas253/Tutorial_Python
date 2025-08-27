## Designing Models in Django

### Purpose

Models in Django represent the **data structure** of your application, mapping to database tables through the **Object-Relational Mapping (ORM)** system.

---

### Considerations Before Designing

* **Entities**: Identify objects in the system (e.g., `User`, `Product`, `Order`).
* **Fields**: Determine attributes for each entity.
* **Data types**: Choose appropriate Django field types.
* **Relationships**: Identify how models relate (One-to-One, One-to-Many, Many-to-Many).
* **Constraints**: Set uniqueness, nullability, default values, and validation rules.
* **Performance**: Plan indexes and query optimization.
* **Extensibility**: Allow for easy future modifications.

---

### Syntax Example

```python
from django.db import models

class Product(models.Model):
    name = models.CharField(
        max_length=100,   # Maximum length of name
        unique=True,      # Must be unique
        null=False,       # Cannot be null in DB
        blank=False       # Cannot be empty in forms
    )
    price = models.DecimalField(
        max_digits=10,    # Total digits allowed
        decimal_places=2, # Digits after decimal
        default=0.00
    )
    description = models.TextField(
        blank=True,       # Optional in forms
        null=True         # Optional in DB
    )
    created_at = models.DateTimeField(auto_now_add=True) # Set on creation
    updated_at = models.DateTimeField(auto_now=True)     # Updated on save

    category = models.ForeignKey(
        'Category',       # Related model
        on_delete=models.CASCADE, # Delete products if category deleted
        related_name='products'   # Reverse lookup name
    )

    class Meta:
        db_table = 'product_table'  # Custom table name
        ordering = ['name']         # Default ordering
        indexes = [models.Index(fields=['price'])] # Index for performance

    def __str__(self):
        return self.name  # String representation
```

---

### Key Model Field Types

* **Basic Fields**: `CharField`, `TextField`, `IntegerField`, `DecimalField`, `FloatField`, `BooleanField`.
* **Date/Time Fields**: `DateField`, `DateTimeField`, `TimeField`, `DurationField`.
* **File/Media Fields**: `FileField`, `ImageField`.
* **Relational Fields**: `ForeignKey`, `OneToOneField`, `ManyToManyField`.
* **Special Fields**: `EmailField`, `URLField`, `UUIDField`, `SlugField`.

---

### Relationships

* **One-to-One**: `OneToOneField` → unique link between two models.
* **One-to-Many**: `ForeignKey` → one record relates to many others.
* **Many-to-Many**: `ManyToManyField` → multiple records relate to multiple others.

---

### Model Methods

* **`__str__()`**: Human-readable object representation.
* **`save()`**: Override to customize save behavior.
* **`get_absolute_url()`**: Returns canonical URL for the object.
* **Custom methods**: For business logic.

---

### Model Meta Options

* `db_table`: Custom table name.
* `ordering`: Default query ordering.
* `unique_together`: Composite uniqueness.
* `index_together`: Composite indexes.
* `verbose_name` / `verbose_name_plural`: Human-readable names.

---

### Workflow of Designing a Model

1. Define model class in `models.py`.
2. Configure fields, relationships, constraints.
3. Add Meta options for database behavior.
4. Create migration:

   ```bash
   python manage.py makemigrations
   ```
5. Apply migration:

   ```bash
   python manage.py migrate
   ```
6. Use the model in queries, forms, and admin.

---
