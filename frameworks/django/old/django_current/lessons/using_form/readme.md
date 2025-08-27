## Using Forms in Django

### Purpose

* Process user input securely.
* Automate validation, data binding, and rendering.

---

### Steps to Use a Form

#### 1. **Import and Instantiate**

```python
from .forms import MyForm

# GET request → empty form
form = MyForm()

# POST request → bind data
form = MyForm(request.POST)
```

---

#### 2. **Validate Form**

```python
if form.is_valid():
    cleaned_data = form.cleaned_data  # Access validated data
```

* `is_valid()` runs all validation rules.
* `cleaned_data` contains safe, processed values.

---

#### 3. **Handle Valid / Invalid Data**

```python
if form.is_valid():
    # Save to DB or process data
    form.save()  # Works for ModelForms
else:
    # Form has errors → will display automatically in template
    print(form.errors)
```

---

#### 4. **Render in Template**

```html
<form method="post">
    {% csrf_token %}
    {{ form.as_p }}  <!-- as_p, as_ul, or custom rendering -->
    <button type="submit">Submit</button>
</form>
```

---

#### 5. **Files in Forms**

```python
form = MyForm(request.POST, request.FILES)  # Include request.FILES for file uploads
```

* Requires `enctype="multipart/form-data"` in `<form>` tag.

---

### Usage Scenarios

* Contact forms
* Registration / login
* Search bars
* Upload forms
* Filtering data

---
