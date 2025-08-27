## Django Admin 

---

### Purpose

Django Admin is a built-in, auto-generated, web-based interface for managing database content via models.

---

### Enabling Admin Interface

1. **Add `'django.contrib.admin'`** in `INSTALLED_APPS` (default).
2. **Ensure these are in `urls.py`:**

```python
from django.contrib import admin
from django.urls import path

urlpatterns = [
    path('admin/', admin.site.urls),
]
```

3. **Run migrations** (if not already):

```bash
python manage.py migrate
```

4. **Create superuser**:

```bash
python manage.py createsuperuser
```

5. **Visit** `http://127.0.0.1:8000/admin`

---

### Registering Models

In `admin.py`:

```python
from django.contrib import admin
from .models import MyModel

admin.site.register(MyModel)
```

---

### Customizing Admin Interface

#### Using `ModelAdmin` class

```python
class MyModelAdmin(admin.ModelAdmin):
    list_display = ('field1', 'field2')
    list_filter = ('status',)
    search_fields = ('name', 'email')
    ordering = ('-created_at',)
    readonly_fields = ('created_at',)
    list_editable = ('status',)

admin.site.register(MyModel, MyModelAdmin)
```

#### Additional Options

| Option            | Description                       |
| ----------------- | --------------------------------- |
| `list_display`    | Columns to show in list view      |
| `list_filter`     | Sidebar filters                   |
| `search_fields`   | Enables search bar                |
| `ordering`        | Default ordering                  |
| `readonly_fields` | Fields that can't be edited       |
| `list_editable`   | Allow inline editing in list view |
| `fieldsets`       | Group fields in form              |
| `exclude`         | Exclude fields from the form      |

---

### Inline Models

Display related models inside the admin:

```python
class BookInline(admin.TabularInline):  # or admin.StackedInline
    model = Book

class AuthorAdmin(admin.ModelAdmin):
    inlines = [BookInline]
```

---

### Custom Actions

```python
@admin.action(description="Mark selected items as published")
def make_published(modeladmin, request, queryset):
    queryset.update(status='published')

class PostAdmin(admin.ModelAdmin):
    actions = [make_published]
```

---

### Custom Admin Templates & CSS

Override `admin/base_site.html`, `admin/index.html`, etc., in a folder:

```
templates/admin/
```

Add custom static CSS/JS via `Media` class in `ModelAdmin`:

```python
class MyModelAdmin(admin.ModelAdmin):
    class Media:
        css = {
            'all': ['my_styles.css']
        }
        js = ['my_script.js']
```

---

### Permissions and User Access

* Add user via Admin → set permissions
* Control model-level access using `ModelAdmin.has_add_permission`, `has_change_permission`, etc.

---
