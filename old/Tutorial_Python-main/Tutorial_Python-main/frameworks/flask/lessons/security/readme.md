## Security in Flask  

### Overview  
Flask applications must be secured against **common vulnerabilities** such as **SQL Injection, XSS, CSRF, authentication issues, and improper configurations**. Below are key security practices.  

---

## 1. **Input Validation & SQL Injection Prevention**  
### Use ORM (Flask-SQLAlchemy)  
```python
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
```
🚫 **Avoid raw SQL queries** like:  
```python
db.session.execute(f"SELECT * FROM users WHERE username = '{username}'")  # UNSAFE
```
✅ **Use parameterized queries**:  
```python
db.session.execute("SELECT * FROM users WHERE username = :username", {"username": username})
```

---

## 2. **Cross-Site Scripting (XSS) Protection**  
🚫 **Avoid rendering user input directly**:  
```html
<div>{{ user_input }}</div>  <!-- UNSAFE -->
```
✅ **Use Jinja2 auto-escaping** (enabled by default):  
```html
<div>{{ user_input | e }}</div>
```
✅ **Sanitize input** with `bleach`:  
```sh
pip install bleach
```
```python
import bleach
safe_input = bleach.clean(user_input)
```

---

## 3. **Cross-Site Request Forgery (CSRF) Protection**  
### Use Flask-WTF  
```sh
pip install flask-wtf
```
```python
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired
from flask_wtf.csrf import CSRFProtect

app.config["SECRET_KEY"] = "your_secret_key"
csrf = CSRFProtect(app)

class MyForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
```

✅ **Include CSRF token in forms**:  
```html
<form method="POST">
    {{ form.hidden_tag() }}
    <input type="text" name="name">
    <input type="submit">
</form>
```

---

## 4. **Authentication & Password Hashing**  
### Use Flask-Login and Flask-Bcrypt  
```sh
pip install flask-login flask-bcrypt
```
```python
from flask_login import UserMixin
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)
```

---

## 5. **Session Security & Cookie Protection**  
```python
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SECURE"] = True  # Use HTTPS
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
```
🚫 **Avoid storing sensitive data in cookies**.  

---

## 6. **Rate Limiting (Prevent Brute Force Attacks)**  
```sh
pip install flask-limiter
```
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per minute"])

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")
def login():
    return "Login attempt"
```

---

## 7. **Secure Headers (Using Flask-Talisman)**  
```sh
pip install flask-talisman
```
```python
from flask_talisman import Talisman

Talisman(app, content_security_policy=None)
```
✅ **Enables HTTPS, HSTS, and X-Frame-Options**  

---

## Summary  

| Security Measure | Description |
|-----------------|------------|
| **SQL Injection Prevention** | Use ORM (SQLAlchemy) and parameterized queries |
| **XSS Prevention** | Use Jinja2 auto-escaping and sanitize input |
| **CSRF Protection** | Use `flask-wtf` and CSRF tokens |
| **Password Security** | Use Flask-Bcrypt for hashing |
| **Session Security** | Secure cookies with `SESSION_COOKIE_SECURE=True` |
| **Rate Limiting** | Prevent brute-force attacks with Flask-Limiter |
| **Secure Headers** | Use Flask-Talisman for security headers |
