
## Django Template Inheritance   

Template inheritance allows you to create a **base (parent) template** that defines a general structure and then **extend it in child templates** by overriding specific blocks.

---

### 1. **Core Concepts**

| Concept            | Description                                                                |
| ------------------ | -------------------------------------------------------------------------- |
| Base Template      | The main layout (e.g., header, footer, layout). Stored in `templates/`.    |
| Child Template     | Extends the base and overrides or fills `block` sections.                  |
| Block              | A named placeholder in the base template.                                  |
| `{% extends %}`    | Declares the parent template being extended.                               |
| `{% block name %}` | Used to define named content areas in the base and override them in child. |
| `{% endblock %}`   | Marks the end of a block.                                                  |

---

### 2. **Base Template Example Structure**

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html>
<head>
  <title>{% block title %}My Site{% endblock %}</title>
</head>
<body>
  <div class="header">{% block header %}{% endblock %}</div>
  <div class="content">{% block content %}{% endblock %}</div>
  <div class="footer">{% block footer %}{% endblock %}</div>
</body>
</html>
```

---

### 3. **Child Template**

```html
<!-- templates/home.html -->
{% extends "base.html" %}

{% block title %}Home{% endblock %}

{% block content %}
  <h1>Welcome to Home Page</h1>
{% endblock %}
```

---

### 4. **Block Rules**

* Blocks can be **empty or pre-filled** in the base template.
* Child templates can **override any block** or leave them as-is.
* You can **nest block inheritance** across multiple templates.

---

### 5. **Use Cases**

| Use Case           | Example Files                           | Purpose                                  |
| ------------------ | --------------------------------------- | ---------------------------------------- |
| Single layout site | `base.html`, `home.html`, `about.html`  | Maintain consistent layout               |
| Multi-layout site  | `base.html`, `dashboard_base.html`      | Use different bases for admin/user views |
| App-specific bases | `products/base.html`, `users/base.html` | Each app can have its own base layout    |

---

### 6. **Best Practices**

* Keep `base.html` minimal and clean.
* Use separate templates for layouts (`admin_base.html`, `public_base.html`).
* Define commonly used blocks: `title`, `content`, `scripts`, `sidebar`, etc.
* Avoid deeply nested inheritance trees.

---

### 7. **Additional Tags for Extending**

| Tag                 | Purpose                                        |
| ------------------- | ---------------------------------------------- |
| `{% include %}`     | Includes another template (like partial views) |
| `{% block.super %}` | Appends parent block’s content inside override |
| `{% load static %}` | Loads static files like CSS/JS                 |

---

### 8. **Summary Table**

| Feature            | Tag                    | Scope                     |
| ------------------ | ---------------------- | ------------------------- |
| Extend a base file | `{% extends "file" %}` | At top of child template  |
| Define placeholder | `{% block name %}`     | In base template          |
| Override content   | `{% block name %}`     | In child template         |
| Use parent content | `{{ block.super }}`    | Inside overridden block   |
| Partial templates  | `{% include "file" %}` | Anywhere inside templates |

---
