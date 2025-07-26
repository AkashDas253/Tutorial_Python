## Function-Based Views (FBV) 

---

### What are FBVs?

* Views defined as **Python functions**.
* Take a `request` object as argument.
* Return an `HttpResponse` or subclass.
* Focused, simple, and explicit.

---

### Basic Syntax

```python
from django.http import HttpResponse

def my_view(request):
    return HttpResponse("Hello from FBV")
```

---

### Structure

```python
def view_name(request, *args, **kwargs):
    # Process logic
    return HttpResponse / render / redirect
```

---

### Common Return Functions

| Function         | Purpose                      |
| ---------------- | ---------------------------- |
| `HttpResponse()` | Return plain text or HTML    |
| `render()`       | Render template with context |
| `redirect()`     | Redirect to another URL      |
| `JsonResponse()` | Return JSON response         |

---

### Accessing Data in FBVs

* **GET data:** `request.GET.get('key')`
* **POST data:** `request.POST.get('key')`
* **Files:** `request.FILES['file']`
* **Session:** `request.session['key']`
* **User:** `request.user`

---

### Template Rendering

```python
from django.shortcuts import render

def home(request):
    context = {'title': 'Homepage'}
    return render(request, 'home.html', context)
```

---

### Handling Forms

```python
def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # process form
            return redirect('success')
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})
```

---

### Using Decorators

| Decorator                 | Purpose                        |
| ------------------------- | ------------------------------ |
| `@login_required`         | Require login                  |
| `@require_http_methods()` | Restrict to GET/POST/etc.      |
| `@csrf_exempt`            | Disable CSRF (not recommended) |

```python
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')
```

---

### URL Mapping

```python
from django.urls import path
from .views import my_view

urlpatterns = [
    path('hello/', my_view, name='hello'),
]
```

---

### Advantages of FBV

* Simple and direct
* More readable for small tasks
* Easy to control flow manually

---

### Limitations of FBV

* Less reusable than CBVs
* Repetition in CRUD logic
* Harder to extend via inheritance

---
