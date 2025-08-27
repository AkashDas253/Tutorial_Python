### CRUD Views (Function-Based Views) in Django

---

#### **Create View**

```python
from django.shortcuts import render, redirect
from .forms import MyModelForm

def create_item(request):
    if request.method == 'POST':
        form = MyModelForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('item_list')  # Named URL
    else:
        form = MyModelForm()
    return render(request, 'item_form.html', {'form': form})
```

---

#### **Read/List View**

```python
from django.shortcuts import render
from .models import MyModel

def item_list(request):
    items = MyModel.objects.all()
    return render(request, 'item_list.html', {'items': items})
```

---

#### **Detail View**

```python
from django.shortcuts import render, get_object_or_404
from .models import MyModel

def item_detail(request, pk):
    item = get_object_or_404(MyModel, pk=pk)
    return render(request, 'item_detail.html', {'item': item})
```

---

#### **Update View**

```python
from django.shortcuts import render, redirect, get_object_or_404
from .forms import MyModelForm
from .models import MyModel

def update_item(request, pk):
    item = get_object_or_404(MyModel, pk=pk)
    if request.method == 'POST':
        form = MyModelForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            return redirect('item_list')
    else:
        form = MyModelForm(instance=item)
    return render(request, 'item_form.html', {'form': form})
```

---

#### **Delete View**

```python
from django.shortcuts import render, redirect, get_object_or_404
from .models import MyModel

def delete_item(request, pk):
    item = get_object_or_404(MyModel, pk=pk)
    if request.method == 'POST':
        item.delete()
        return redirect('item_list')
    return render(request, 'item_confirm_delete.html', {'item': item})
```

---
