## Blueprints in Flask  

### Overview  
- **Blueprints** allow modularizing a Flask application into multiple components.  
- Useful for large applications to keep routes, views, and logic organized.  

---

## Creating a Blueprint  

### Step 1: Define a Blueprint  
Create a separate Python file (e.g., `routes.py`) and define a blueprint.

```python
from flask import Blueprint

# Define a blueprint
bp = Blueprint('main', __name__)

@bp.route('/')
def home():
    return "Welcome to Flask Blueprints!"

@bp.route('/about')
def about():
    return "This is the about page."
```

---

### Step 2: Register the Blueprint  
In the main application file (e.g., `app.py`), import and register the blueprint.

```python
from flask import Flask
from routes import bp  # Import the blueprint

app = Flask(__name__)
app.register_blueprint(bp)  # Register the blueprint

if __name__ == '__main__':
    app.run(debug=True)
```

- `app.register_blueprint(bp)` integrates the blueprint into the main app.  

---

## Using Blueprints for Modular Apps  

### Example: Organizing Routes by Functionality  
📂 **Project Structure**  
```
/flask_app
│── app.py
│── /blueprints
│   │── __init__.py
│   │── users.py
│   │── products.py
```

### **users.py** (User-related routes)
```python
from flask import Blueprint

users_bp = Blueprint('users', __name__, url_prefix='/users')

@users_bp.route('/profile')
def profile():
    return "User Profile Page"
```

---

### **products.py** (Product-related routes)
```python
from flask import Blueprint

products_bp = Blueprint('products', __name__, url_prefix='/products')

@products_bp.route('/list')
def product_list():
    return "Product List Page"
```

---

### **app.py** (Main Application)
```python
from flask import Flask
from blueprints.users import users_bp
from blueprints.products import products_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(users_bp)
app.register_blueprint(products_bp)

if __name__ == '__main__':
    app.run(debug=True)
```

Now, the app supports:  
- `/users/profile` → User profile  
- `/products/list` → Product list  

---

## Summary  

| Feature | Description |
|---------|------------|
| **Blueprints** | Modularize a Flask app into reusable components |
| **Defining a Blueprint** | `Blueprint('name', __name__)` |
| **Registering a Blueprint** | `app.register_blueprint(bp)` |
| **Organizing Routes** | Group related views into separate files |
| **URL Prefix** | `url_prefix='/section'` for grouping routes |
