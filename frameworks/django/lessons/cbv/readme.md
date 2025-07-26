## 📘 Class-Based Views (CBVs) 

---

### 🔹 What are CBVs?

* Django views defined as **classes** instead of functions.
* Provide **reusable**, **extensible**, and **organized** ways to handle views.
* Can use **mixins**, **inheritance**, and **custom logic** easily.

---

### 🔹 Basic Syntax

```python
from django.views import View
from django.http import HttpResponse

class MyView(View):
    def get(self, request):
        return HttpResponse('Hello from CBV')
```

---

### 🔹 Routing CBVs

```python
from django.urls import path
from .views import MyView

urlpatterns = [
    path('hello/', MyView.as_view(), name='hello'),
]
```

> **Important:** Always use `.as_view()` to bind the CBV to a URL.

---

### 🔹 Method Handlers

| Method      | Purpose                  |
| ----------- | ------------------------ |
| `get()`     | Handles GET requests     |
| `post()`    | Handles POST requests    |
| `put()`     | Handles PUT requests     |
| `delete()`  | Handles DELETE requests  |
| `patch()`   | Handles PATCH requests   |
| `head()`    | Handles HEAD requests    |
| `options()` | Handles OPTIONS requests |

---

### 🔹 Template Rendering

```python
from django.shortcuts import render
from django.views import View

class HomeView(View):
    def get(self, request):
        context = {'title': 'Home'}
        return render(request, 'home.html', context)
```

---

### 🔹 Class-Based View Categories

#### 📂 **Base CBVs (Low-level)**

| View Class     | Description                   |
| -------------- | ----------------------------- |
| `View`         | Base class for all CBVs       |
| `TemplateView` | Renders template with context |
| `RedirectView` | Redirects to another URL      |

#### 📂 **Generic Display Views**

| View Class   | Description             |
| ------------ | ----------------------- |
| `DetailView` | Display a single object |
| `ListView`   | Display list of objects |

#### 📂 **Generic Editing Views**

| View Class   | Description               |
| ------------ | ------------------------- |
| `CreateView` | Create a new object       |
| `UpdateView` | Update an existing object |
| `DeleteView` | Delete an object          |
| `FormView`   | Handle custom form logic  |

---

### 🔹 Mixin Classes

Used to add modular functionality:

| Mixin                     | Purpose                      |
| ------------------------- | ---------------------------- |
| `LoginRequiredMixin`      | Require user to be logged in |
| `PermissionRequiredMixin` | Check user permissions       |
| `UserPassesTestMixin`     | Custom user logic            |
| `SuccessMessageMixin`     | Show success message on save |
| `ContextMixin`            | Add context to templates     |

---

### 🔹 Example: `ListView` Usage

```python
from django.views.generic import ListView
from .models import Product

class ProductListView(ListView):
    model = Product
    template_name = 'products.html'
    context_object_name = 'products'
```

---

### 🔹 Example: `CreateView` Usage

```python
from django.views.generic.edit import CreateView
from .models import Contact
from django.urls import reverse_lazy

class ContactCreateView(CreateView):
    model = Contact
    fields = ['name', 'email', 'message']
    template_name = 'contact.html'
    success_url = reverse_lazy('thank_you')
```

---

### 🔹 Advantages of CBVs

* Code reuse via inheritance
* Modular via mixins
* Cleaner and scalable
* Easy CRUD with minimal code

---

### 🔹 Disadvantages of CBVs

* Higher learning curve
* Overhead for very simple views
* Can be confusing due to inheritance

---
