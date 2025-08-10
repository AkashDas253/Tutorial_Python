## Flask Cheatsheet

This cheatsheet covers all key concepts, syntax, and usage patterns in Flask for quick reference during development.

---

### 🏗️ Project Structure

```
/project
│
├── app.py              # Main Flask app
├── /templates          # HTML templates
│   └── index.html
├── /static             # CSS, JS, images
├── /instance           # Config or database files
└── /venv               # Virtual environment (optional)
```

---

### 🚀 Basic App Setup

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
```

---

### 🧭 Routing

```python
@app.route('/')               # Basic route
@app.route('/user/<name>')    # Variable rule
@app.route('/post/<int:id>')  # Type-specific route
@app.route('/path/<path:p>')  # Catch-all path
def greet(name=None):
    return f'Hello {name}'
```

---

### 📥 Request Methods

```python
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        data = request.form['key']
```

---

### 📦 Request & Response Objects

```python
from flask import request, jsonify, make_response

request.args        # GET params
request.form        # POST form data
request.json        # JSON data
request.files       # Uploaded files

make_response()     # Custom response
jsonify(data)       # Return JSON
```

---

### 🧾 Templates (Jinja2)

```python
from flask import render_template

@app.route('/')
def index():
    return render_template('index.html', name='User')
```

**index.html (Jinja2)**

```html
<h1>Hello, {{ name }}!</h1>
{% if name %}
  <p>Welcome back!</p>
{% endif %}
```

---

### 🗂️ Static Files

```html
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
```

---

### 🛠️ Forms

```python
# HTML form
<form method="POST" action="/submit">
  <input name="username">
  <button type="submit">Submit</button>
</form>
```

---

### 🧠 Sessions

```python
from flask import session

app.secret_key = 'your_secret'

session['user'] = 'Alice'       # Set session
user = session.get('user')      # Get session
session.pop('user', None)       # Delete session
```

---

### 🔒 Redirects & URL Building

```python
from flask import redirect, url_for

@app.route('/login')
def login():
    return redirect(url_for('dashboard'))
```

---

### 📄 File Uploads

```python
file = request.files['file']
file.save('uploads/' + file.filename)
```

---

### 🧰 Blueprints (Modular Apps)

```python
# users.py
from flask import Blueprint

users = Blueprint('users', __name__)

@users.route('/profile')
def profile():
    return 'User Profile'

# Register
app.register_blueprint(users, url_prefix='/users')
```

---

### 🧩 Error Handling

```python
@app.errorhandler(404)
def not_found(e):
    return 'Page Not Found', 404
```

---

### 🛡️ Configuration

```python
app.config['DEBUG'] = True
app.config.from_pyfile('config.py')
```

---

### 🐍 CLI Commands

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

---

### 🧪 Testing

```python
import unittest

class MyTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
```

---

### 🛢️ Database (Using Flask-SQLAlchemy)

```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

db.create_all()
```

---

### 🧩 Extensions

| Extension           | Use                           |
|---------------------|-------------------------------|
| Flask-SQLAlchemy     | ORM support                   |
| Flask-WTF            | Form handling and CSRF        |
| Flask-Migrate        | DB migrations                 |
| Flask-Login          | User authentication           |
| Flask-Mail           | Sending emails                |
| Flask-Caching        | Response caching              |

---
