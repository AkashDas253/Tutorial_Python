## Routing in Flask  

### Overview  
Routing in Flask is the mechanism that maps URLs to view functions, allowing the application to handle different requests. Flask uses decorators to define routes.

---

### Basic Route  
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Flask!"
```
- `@app.route('/')` → Maps the root URL (`/`) to the `home` function.  
- The function returns a response (string, HTML, or template).  

---

### Route with Multiple URLs  
A function can be mapped to multiple routes.  
```python
@app.route('/about')
@app.route('/info')
def about():
    return "This is the About page."
```
- Both `/about` and `/info` will trigger `about()`.

---

### Route with Variable URL (Dynamic Routing)  
```python
@app.route('/user/<username>')
def user_profile(username):
    return f"Hello, {username}!"
```
- `<username>` acts as a placeholder.  
- Accessed via `/user/Alice` → Output: `"Hello, Alice!"`.  

#### Variable Types  
Flask supports data type conversion in URLs:  
```python
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f"Post ID: {post_id}"
```
| Type | Example |
|------|---------|
| `<string:var>` | `/hello/Alice` |
| `<int:var>` | `/post/5` |
| `<float:var>` | `/price/19.99` |
| `<path:var>` | `/docs/api/v1` (captures entire path) |

---

### Handling Methods (GET, POST, etc.)  
Routes can handle different HTTP methods.  
```python
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        return "Form submitted!"
    return "Submit a form."
```
- Default method is `GET`.  
- Allows `POST` requests for form handling.  

---

### Redirecting Routes  
Use `redirect()` to send users to another URL.  
```python
from flask import redirect, url_for

@app.route('/old')
def old_page():
    return redirect(url_for('home'))
```
- `/old` redirects to `/`.  

---

### URL Building with `url_for()`  
Generates URLs dynamically.  
```python
@app.route('/profile/<username>')
def profile(username):
    return f"Profile of {username}"

@app.route('/link')
def link():
    return url_for('profile', username='Alice')
```
- `/link` returns `'/profile/Alice'`.  

---

### Blueprints (Modular Routing)  
Used to organize routes in large applications.  

**Structure:**  
```
/myapp
  ├── app/
  │   ├── __init__.py
  │   ├── routes.py
  │   ├── auth/
  │   │   ├── __init__.py
  │   │   ├── routes.py
  ├── run.py
```

**`auth/routes.py` (Blueprint Example)**  
```python
from flask import Blueprint

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    return "Login Page"
```

**`app/__init__.py` (Register Blueprint)**  
```python
from flask import Flask
from app.auth.routes import auth

app = Flask(__name__)
app.register_blueprint(auth, url_prefix='/auth')
```
- `/auth/login` is now available.  

---

### Summary  
| Feature | Description |
|---------|------------|
| **Basic Route** | Maps a URL to a function |
| **Multiple Routes** | One function handles multiple URLs |
| **Dynamic Routes** | Handles URL parameters (`<int:id>`, `<string:name>`) |
| **Methods** | Supports `GET`, `POST`, etc. |
| **Redirects** | Redirect users to another page |
| **URL Building** | Generates URLs dynamically using `url_for()` |
| **Blueprints** | Modular structure for large applications |
