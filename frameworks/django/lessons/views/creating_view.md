## Creating Views in Django

---

### Purpose of Views

* Views process web requests and return web responses.
* In Django, views connect models, templates, and business logic.

---

### Types of Views

#### Function-Based Views (FBV)

* Traditional way using Python functions.

```python
from django.http import HttpResponse

def my_view(request):
    return HttpResponse("Hello, world!")
```

#### Class-Based Views (CBV)

* Uses OOP principles; reusable, extendable.

```python
from django.views import View
from django.http import HttpResponse

class MyView(View):
    def get(self, request):
        return HttpResponse("Hello from CBV")
```

#### Generic Class-Based Views

* Provides pre-built views for common patterns.

```python
from django.views.generic import ListView, DetailView
from .models import Post

class PostListView(ListView):
    model = Post
    template_name = "post_list.html"
```

| Generic View | Description                   |
| ------------ | ----------------------------- |
| `ListView`   | List objects                  |
| `DetailView` | Show detail of object         |
| `CreateView` | Form to create object         |
| `UpdateView` | Form to update object         |
| `DeleteView` | Form to confirm delete object |

---

### Return Formats

| Return Type              | Description                   |
| ------------------------ | ----------------------------- |
| `HttpResponse()`         | Raw HTML/text                 |
| `JsonResponse()`         | JSON data (for APIs)          |
| `render()`               | Renders template with context |
| `redirect()`             | Redirects to another URL/view |
| `HttpResponseNotFound()` | 404 response                  |
| `HttpResponseRedirect()` | Alternative redirect          |

---

### View Decorators (FBV only, or with method decorators in CBV)

| Decorator               | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `@login_required`       | Ensures user is authenticated            |
| `@permission_required`  | Requires specific permission             |
| `@require_http_methods` | Restrict to GET/POST/etc.                |
| `@csrf_exempt`          | Disables CSRF protection (use carefully) |
| `@cache_page`           | Cache the view's output                  |

---

### Template Rendering

```python
from django.shortcuts import render

def home(request):
    return render(request, "home.html", {"key": "value"})
```

---

### URL Mapping

* Must map view in `urls.py` to make it accessible.

```python
from django.urls import path
from .views import home

urlpatterns = [
    path('', home, name='home'),
]
```

---

### Using Views for APIs

* Use `JsonResponse()` for data output.

```python
from django.http import JsonResponse

def data_view(request):
    return JsonResponse({"msg": "hello"})
```

* Use Django REST Framework (DRF) for full APIs (beyond base Django).

---

### Best Practices

* Use CBVs for reusable patterns, FBVs for simplicity.
* Use decorators for access control.
* Keep views clean; offload logic to services/helpers.
* Use context dictionaries to pass data to templates.
* Modularize with `views.py`, `api_views.py`, `admin_views.py`, etc.

---
