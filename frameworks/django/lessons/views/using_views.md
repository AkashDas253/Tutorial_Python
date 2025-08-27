## Using Views in Django

---

### Purpose of Views

* Views are Python functions or classes that receive web requests and return web responses.
* Core responsibility: **business logic**, **rendering templates**, **returning HTTP responses**.

---

### View Categories

| Type                                 | Description                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| **Function-Based Views (FBV)**       | Use simple Python functions to handle requests.                  |
| **Class-Based Views (CBV)**          | Use Python classes with inheritance to handle views modularly.   |
| **Generic Class-Based Views (GCBV)** | Provide common patterns like `ListView`, `DetailView` pre-built. |
| **Async Views (Django 3.1+)**        | Support async-def for non-blocking I/O operations.               |

---

### Function-Based Views (FBV)

```python
from django.http import HttpResponse

def my_view(request):
    return HttpResponse("Hello, world!")
```

* Pros: Simple, explicit
* Cons: Harder to scale and reuse

---

### Class-Based Views (CBV)

```python
from django.views import View
from django.http import HttpResponse

class MyView(View):
    def get(self, request):
        return HttpResponse("Hello from class-based view")
```

* Uses HTTP method handlers (`get`, `post`, etc.)
* Inheritance allows code reuse

---

### Generic Class-Based Views (GCBV)

```python
from django.views.generic import ListView
from .models import Book

class BookListView(ListView):
    model = Book
    template_name = 'book_list.html'
```

| View Class     | Purpose                   |
| -------------- | ------------------------- |
| `ListView`     | Display a list of objects |
| `DetailView`   | Display a single object   |
| `CreateView`   | Create a new object       |
| `UpdateView`   | Update an object          |
| `DeleteView`   | Delete an object          |
| `TemplateView` | Render a static template  |
| `RedirectView` | Redirect to another URL   |

---

### Async Views (Django 3.1+)

```python
import asyncio
from django.http import JsonResponse

async def my_async_view(request):
    await asyncio.sleep(1)
    return JsonResponse({'status': 'done'})
```

---

### View Decorators

| Decorator                                | Purpose                              |
| ---------------------------------------- | ------------------------------------ |
| `@login_required`                        | Require user to be authenticated     |
| `@require_http_methods(["POST"])`        | Limit to specific methods            |
| `@csrf_exempt`                           | Disable CSRF protection for the view |
| `@permission_required('app.permission')` | Require specific permission          |
| `@user_passes_test`                      | Apply custom user test logic         |
| `@cache_page(60 * 15)`                   | Cache the view output for 15 minutes |
| `@never_cache`                           | Prevent caching of view              |
| `@vary_on_headers` / `@vary_on_cookie`   | Add Vary headers for caching control |

---

### Return Types from Views

| Return Type                          | Description                    |
| ------------------------------------ | ------------------------------ |
| `HttpResponse()`                     | Basic plain-text/html response |
| `JsonResponse()`                     | Return JSON response           |
| `HttpResponseRedirect()`             | Redirect to another URL        |
| `render(request, template, context)` | Return rendered template       |
| `redirect()`                         | Shortcut for redirection       |
| `Http404`                            | Raise 404 error                |
| `FileResponse()`                     | Serve files like PDFs, images  |

---

### Common View Concepts

* `request.method` → GET, POST, etc.
* `request.GET` / `request.POST` → Query parameters
* `request.user` → Logged-in user
* `request.FILES` → File uploads
* `request.session` → Session data
* `request.is_ajax()` → AJAX request detection (deprecated in Django 4+)

---

### Error and Exception Handling

```python
from django.http import Http404
from django.shortcuts import get_object_or_404

def detail(request, pk):
    obj = get_object_or_404(MyModel, pk=pk)
    ...
```

* `Http404` to raise "Not Found"
* Custom 404, 403, 500 templates (`404.html`, etc.)

---

### Best Practices

* Use CBVs for modularity and reuse.
* Use GCBVs when working with models for CRUD operations.
* Decorate FBVs with `@login_required`, `@csrf_exempt`, etc.
* Use `get_context_data()` to add extra context to CBVs.
* Keep views thin—delegate logic to services/forms/models where possible.

---
