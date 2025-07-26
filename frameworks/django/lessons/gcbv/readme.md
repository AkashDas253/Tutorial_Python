## Generic Class-Based Views (GCBVs) 

---

### 🔹 What Are GCBVs?

* GCBVs are **pre-built views** for common patterns like listing, creating, updating, or deleting objects.
* Built on top of `View`, `TemplateResponseMixin`, `ContextMixin`, and others.
* Save time and reduce boilerplate.

---

### 🔹 Structure of a GCBV

```python
class MyView(GenericViewClass):
    model = MyModel
    template_name = 'template.html'
    context_object_name = 'object'
    success_url = reverse_lazy('some-url')  # for Create/Update/Delete
```

---

### 🔹 Display Views (Read)

| View Class   | Purpose                    | Key Attributes                                                        |
| ------------ | -------------------------- | --------------------------------------------------------------------- |
| `ListView`   | Displays a list of objects | `model`, `context_object_name`, `paginate_by`, `ordering`, `queryset` |
| `DetailView` | Displays a single object   | `model`, `pk_url_kwarg`, `slug_field`, `context_object_name`          |

---

### 🔹 Editing Views (Create/Update/Delete)

| View Class   | Purpose                      | Key Attributes                                 |
| ------------ | ---------------------------- | ---------------------------------------------- |
| `CreateView` | Handles object creation form | `model`, `form_class`, `fields`, `success_url` |
| `UpdateView` | Handles editing an object    | same as `CreateView`                           |
| `DeleteView` | Handles object deletion      | `model`, `success_url`, `template_name`        |

---

### 🔹 Form Handling View

| View Class | Purpose                  | Key Attributes                                                                         |
| ---------- | ------------------------ | -------------------------------------------------------------------------------------- |
| `FormView` | Uses a custom form class | `form_class`, `success_url`, `template_name`, `get_form`, `form_valid`, `form_invalid` |

---

### 🔹 Required Mixins (Behind the Scenes)

| Mixin                   | Role                                   |
| ----------------------- | -------------------------------------- |
| `SingleObjectMixin`     | Used in `DetailView`, `UpdateView`     |
| `MultipleObjectMixin`   | Used in `ListView`                     |
| `FormMixin`             | Used in `FormView`, `CreateView`, etc. |
| `ModelFormMixin`        | Adds model form capabilities           |
| `ProcessFormView`       | Adds form handling logic               |
| `TemplateResponseMixin` | Adds `render_to_response`              |

---

### 🔹 Example: `ListView`

```python
from django.views.generic import ListView
from .models import Product

class ProductListView(ListView):
    model = Product
    template_name = 'products.html'
    context_object_name = 'products'
    paginate_by = 10
```

---

### 🔹 Example: `CreateView`

```python
from django.views.generic.edit import CreateView
from .models import Contact
from django.urls import reverse_lazy

class ContactCreateView(CreateView):
    model = Contact
    fields = ['name', 'email', 'message']
    template_name = 'contact.html'
    success_url = reverse_lazy('thanks')
```

---

### 🔹 Overriding Key Methods

| Method               | Purpose                     |
| -------------------- | --------------------------- |
| `get_queryset()`     | Customize queryset          |
| `get_context_data()` | Add extra context           |
| `form_valid()`       | Custom success behavior     |
| `form_invalid()`     | Custom error behavior       |
| `get_success_url()`  | Dynamic success redirection |

---

### 🔹 Inheritance Tree (Simplified – Mermaid)

```mermaid
classDiagram;
    View <|-- TemplateView;
    View <|-- FormView;
    FormView <|-- CreateView;
    FormView <|-- UpdateView;
    View <|-- DeleteView;
    View <|-- ListView;
    View <|-- DetailView;
```

---

### 🔹 Advantages

* Rapid development with less code.
* Built-in form and model handling.
* Easy to extend using mixins.

---

### 🔹 Disadvantages

* Inheritance chain can become hard to trace.
* Overhead for small, simple views.

---
