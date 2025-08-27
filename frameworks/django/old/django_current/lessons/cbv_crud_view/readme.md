## **CRUD views using Class-Based Views (CBV) in Django**:

---

## Create View

```python
from django.views.generic.edit import CreateView
from .models import MyModel

class MyModelCreateView(CreateView):
    model = MyModel
    fields = ['field1', 'field2']  # Model fields to include in the form
    template_name = 'myapp/mymodel_form.html'  # Optional, default is <model>_form.html
    success_url = '/success/'  # Redirect URL after form submission
```

---

## Read (List & Detail) Views

```python
from django.views.generic import ListView, DetailView

class MyModelListView(ListView):
    model = MyModel
    template_name = 'myapp/mymodel_list.html'  # Optional

class MyModelDetailView(DetailView):
    model = MyModel
    template_name = 'myapp/mymodel_detail.html'  # Optional
```

---

## Update View

```python
from django.views.generic.edit import UpdateView

class MyModelUpdateView(UpdateView):
    model = MyModel
    fields = ['field1', 'field2']
    template_name = 'myapp/mymodel_form.html'
    success_url = '/success/'
```

---

## Delete View

```python
from django.views.generic.edit import DeleteView

class MyModelDeleteView(DeleteView):
    model = MyModel
    template_name = 'myapp/mymodel_confirm_delete.html'
    success_url = '/success/'
```

---

## URL Configuration

```python
from django.urls import path
from .views import (
    MyModelListView, MyModelDetailView, 
    MyModelCreateView, MyModelUpdateView, MyModelDeleteView
)

urlpatterns = [
    path('', MyModelListView.as_view(), name='mymodel_list'),
    path('<int:pk>/', MyModelDetailView.as_view(), name='mymodel_detail'),
    path('create/', MyModelCreateView.as_view(), name='mymodel_create'),
    path('<int:pk>/update/', MyModelUpdateView.as_view(), name='mymodel_update'),
    path('<int:pk>/delete/', MyModelDeleteView.as_view(), name='mymodel_delete'),
]
```

---
