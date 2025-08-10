## Named URL in Django

### Concept

* **Named URL**: Assigning a unique name to a URL pattern in Django so it can be referenced throughout the project (templates, views, redirects) without hardcoding the path.
* Helps **avoid breaking links** if the actual path changes, since only the URL name remains consistent.

---

### Declaration in `urls.py`

```python
from django.urls import path
from . import views

urlpatterns = [
    path('article/<int:id>/', views.article_detail, name='article-detail'),  
    # 'name' parameter assigns a unique name to the URL pattern
]
```

---

### Usage in Templates

```html
<a href="{% url 'article-detail' id=5 %}">Read Article</a>
```

---

### Usage in Views (Reverse Resolution)

```python
from django.urls import reverse
from django.shortcuts import redirect

# reverse() returns the full URL path for a given name
url_path = reverse('article-detail', kwargs={'id': 5})

# redirect() can use URL names directly
return redirect('article-detail', id=5)
```

---

### Key Points to Remember

* Defined in `urls.py` using the `name` argument in `path()` or `re_path()`.
* Accessed in templates via `{% url 'name' %}` tag.
* Accessed in Python code via `reverse()` or `redirect()`.
* Works with both **static** and **dynamic** URLs.
* Encouraged for **maintainability** and **DRY principle** compliance.

---
