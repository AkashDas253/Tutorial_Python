## Views in Django

---

### 📌 Purpose of Views

* Views are Python functions or classes that receive web requests and return web responses.
* They **contain the logic** of what data to display and how.

---

### 🧱 Types of Views

| Type                                 | Description                                                                        |
| ------------------------------------ | ---------------------------------------------------------------------------------- |
| **Function-Based Views (FBV)**       | Simple Python functions that take a request and return a response.                 |
| **Class-Based Views (CBV)**          | Organize views using Python classes to support reuse, inheritance, and clean code. |
| **Generic Class-Based Views (GCBV)** | Predefined CBVs for common use-cases (CRUD) with less boilerplate.                 |
| **Async Views**                      | Views using `async def`, suitable for async I/O operations in Django 3.1+          |
| **API Views (via DRF)**              | Used in Django REST Framework for building REST APIs.                              |
| **Decorated Views**                  | FBVs or CBVs enhanced with decorators for authentication, caching, etc.            |

---

### 🧠 Core Concepts

#### 🔹 Request and Response Flow

* Django routes request using `urls.py` to the appropriate view.
* Views interact with:

  * Models → to fetch/update data.
  * Templates → to render responses.
  * HttpResponse → to return plain/text or custom responses.

#### 🔹 View Responsibilities

* Accept `HttpRequest`.
* Execute business logic.
* Fetch or modify data via models.
* Return an `HttpResponse`, `JsonResponse`, `TemplateResponse`, or redirect.

---

### 🔹 Function-Based Views (FBV)

```python
from django.http import HttpResponse

def hello_view(request):
    return HttpResponse("Hello World")
```

* Simple to implement and understand.
* Can be customized using decorators like `@login_required`.

---

### 🔹 Class-Based Views (CBV)

```python
from django.views import View
from django.http import HttpResponse

class HelloView(View):
    def get(self, request):
        return HttpResponse("Hello from CBV")
```

* Encapsulates HTTP method logic inside methods like `.get()`, `.post()` etc.
* Easier to extend and reuse.

---

### 🔹 Generic Class-Based Views (GCBV)

> Pre-built views for CRUD and common tasks.

| View Type      | Purpose                   |
| -------------- | ------------------------- |
| `ListView`     | Display a list of objects |
| `DetailView`   | Display a single object   |
| `CreateView`   | Create new object (form)  |
| `UpdateView`   | Update object (form)      |
| `DeleteView`   | Delete object             |
| `TemplateView` | Render a template only    |
| `RedirectView` | Perform HTTP redirects    |
| `FormView`     | Display and process forms |

> ✅ All extend from `View` and are found in `django.views.generic`.

---

### 🔹 Async Views (Django 3.1+)

```python
from django.http import JsonResponse
import asyncio

async def async_view(request):
    await asyncio.sleep(1)
    return JsonResponse({'status': 'done'})
```

* Use `async def` and `await` for non-blocking operations.

---

### 🔹 Template Rendering in Views

```python
from django.shortcuts import render

def home(request):
    return render(request, "home.html", {"user": "Subham"})
```

---

### 🔹 Redirects and HTTP Response Types

```python
from django.http import HttpResponseRedirect, JsonResponse

def my_view(request):
    return JsonResponse({'key': 'value'})  # JSON
```

---

### 🔹 Decorators for Views

| Decorator                                | Purpose                         |
| ---------------------------------------- | ------------------------------- |
| `@login_required`                        | Restrict access to logged users |
| `@require_http_methods(["GET", "POST"])` | Restrict HTTP methods           |
| `@csrf_exempt`                           | Disable CSRF protection         |

---

### 🔹 Custom Error Views

* Custom handlers for 404, 500, 403, 400:

```python
handler404 = 'myapp.views.custom_404'
```

---

### 🧩 View Utilities

| Utility               | Description                          |
| --------------------- | ------------------------------------ |
| `render()`            | Shortcut to render templates         |
| `redirect()`          | Shortcut to redirect to another view |
| `get_object_or_404()` | Fetch or return 404                  |
| `HttpResponse()`      | Raw response object                  |
| `JsonResponse()`      | JSON output                          |

---

### 🔹 Common Import Paths

| Component               | Path                                          |
| ----------------------- | --------------------------------------------- |
| FBV Tools               | `from django.http`                            |
| CBV & GCBV Base Classes | `from django.views` or `django.views.generic` |
| Shortcuts               | `from django.shortcuts`                       |
| Decorators              | `from django.contrib.auth.decorators`         |

---
