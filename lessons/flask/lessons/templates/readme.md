## Templates in Flask  

### Overview  
Flask uses the **Jinja2** templating engine to generate dynamic HTML content. Templates allow separation of logic and presentation, improving code maintainability.

---

### Setting Up Templates  
- Store HTML templates in the **`templates/`** folder.  
- Use `render_template()` to render templates in Flask routes.  

#### Directory Structure:  
```
/my_flask_app
    /templates
        index.html
        about.html
    app.py
```

---

### Rendering a Template  
#### **app.py**
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
```
- `render_template('index.html')` loads the **index.html** template.

#### **templates/index.html**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Flask App</title>
</head>
<body>
    <h1>Welcome to Flask!</h1>
</body>
</html>
```
- The template renders when accessing `/`.

---

### Passing Data to Templates  
#### **app.py**
```python
@app.route('/user/<name>')
def user(name):
    return render_template('user.html', username=name)
```
#### **templates/user.html**
```html
<h1>Hello, {{ username }}!</h1>
```
- **`{{ username }}`** dynamically displays the passed variable.

---

### Control Structures in Templates  

#### If-Else Condition  
```html
{% if age >= 18 %}
    <p>You are an adult.</p>
{% else %}
    <p>You are a minor.</p>
{% endif %}
```

#### Looping with `for`  
```html
<ul>
{% for item in items %}
    <li>{{ item }}</li>
{% endfor %}
</ul>
```
- Renders a list dynamically.

#### Template Filters  
```html
<p>{{ name | upper }}</p>  <!-- Converts to uppercase -->
```
| Filter | Description |
|--------|------------|
| `upper` | Converts text to uppercase |
| `lower` | Converts text to lowercase |
| `capitalize` | Capitalizes the first letter |
| `length` | Returns the length of an iterable |

---

### Template Inheritance  
Template inheritance allows a base template (`base.html`) to define a common structure.

#### **templates/base.html**
```html
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Flask App{% endblock %}</title>
</head>
<body>
    <header><h1>Website Header</h1></header>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```
- Defines `block title` and `block content` placeholders.

#### **templates/index.html**
```html
{% extends "base.html" %}

{% block title %}Home Page{% endblock %}

{% block content %}
    <h2>Welcome to Flask!</h2>
{% endblock %}
```
- Uses `{% extends "base.html" %}` to inherit from `base.html`.

---

### Including Templates  
Use `{% include "file.html" %}` to reuse components.

#### **templates/header.html**
```html
<header><h1>Welcome to My Website</h1></header>
```

#### **templates/index.html**
```html
{% include "header.html" %}
<p>Main content here.</p>
```
- This helps reuse components across multiple pages.

---

### Summary  

| Feature | Description |
|---------|------------|
| **Basic Rendering** | `render_template('file.html')` loads templates |
| **Passing Data** | Variables are passed using `render_template()` |
| **Control Structures** | `{% if %}`, `{% for %}`, `{% else %}` for logic |
| **Filters** | `upper`, `lower`, `length` for modifying content |
| **Template Inheritance** | `{% extends "base.html" %}` to reuse layouts |
| **Includes** | `{% include "header.html" %}` to insert components |
