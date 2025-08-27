## Rendering Form in Template

### Overview

In Django, rendering a form in a template involves passing a form instance from the view to the template context and using template syntax to display it with proper HTML form tags.

---

### Steps

* **Import and Initialize Form in View**

  ```python
  from django.shortcuts import render
  from .forms import MyForm

  def my_view(request):
      form = MyForm()  # Empty form
      return render(request, 'my_template.html', {'form': form})
  ```

* **Render Form in Template**

  * Basic rendering:

    ```html
    <form method="post">
        {% csrf_token %}
        {{ form }}
        <button type="submit">Submit</button>
    </form>
    ```
  * Render with paragraph tags:

    ```html
    {{ form.as_p }}
    ```
  * Render with table rows:

    ```html
    {{ form.as_table }}
    ```
  * Render as unordered list:

    ```html
    {{ form.as_ul }}
    ```
  * Render fields manually:

    ```html
    <form method="post">
        {% csrf_token %}
        {{ form.field_name.label_tag }}
        {{ form.field_name }}
        {{ form.field_name.errors }}
        <button type="submit">Submit</button>
    </form>
    ```

* **Form Rendering Options**

  * `{{ form }}` → Renders using default layout.
  * `{{ form.as_p }}` → Wraps each field in `<p>`.
  * `{{ form.as_table }}` → Wraps fields in table rows.
  * `{{ form.as_ul }}` → Wraps fields in `<li>`.

---

### Key Points

* Always include `{% csrf_token %}` for POST requests.
* Use manual rendering for custom layouts.
* Field-level rendering allows placing fields anywhere in the HTML.
* Styling can be added with CSS classes in the form widget attributes.

---
