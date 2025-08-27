## Django Form Setup 

### Purpose

Django forms provide a structured way to handle user input, validation, and rendering in HTML, integrating tightly with Django’s model and template layers.

---

### Types of Forms

* **`forms.Form`** – Used for creating custom forms independent of models.
* **`forms.ModelForm`** – Automatically generates form fields from a model’s fields.

---

### Steps to Set Up a Form

#### 1. **Create a Form Class**

In `forms.py` (inside your app):

```python
from django import forms
from .models import MyModel

# Custom form
class MyForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()

# Model-based form
class MyModelForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = '__all__'  # or list of fields ['field1', 'field2']
```

---

#### 2. **Integrate Form in Views**

In `views.py`:

```python
from django.shortcuts import render, redirect
from .forms import MyForm

def my_view(request):
    if request.method == 'POST':
        form = MyForm(request.POST)  # Bind data
        if form.is_valid():
            # Process form.cleaned_data
            return redirect('success')
    else:
        form = MyForm()  # Empty form for GET
    return render(request, 'my_template.html', {'form': form})
```

---

#### 3. **Render Form in Template**

In `my_template.html`:

```html
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}  <!-- as_p / as_table / as_ul -->
    <button type="submit">Submit</button>
</form>
```

---

### Key Considerations

* **CSRF protection**: Always include `{% csrf_token %}` in POST forms.
* **Validation**: Use `form.is_valid()` before accessing `form.cleaned_data`.
* **Widgets**: Customize form field rendering using `widgets` in `forms.Form` or `Meta` of `ModelForm`.
* **Error handling**: Access `form.errors` in templates for user feedback.

---
