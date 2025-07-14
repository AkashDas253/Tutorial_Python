## Flask-Login  

### Overview  
Flask-Login is an authentication extension for Flask that manages user sessions, login states, and access control.

---

## Installation  
Install Flask-Login using pip:  
```sh
pip install flask-login
```

---

## Configuration  
Initialize Flask-Login in the Flask app:  
```python
from flask import Flask
from flask_login import LoginManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for session management

login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect unauthorized users
```

---

## User Model  
Flask-Login requires a **User** model with authentication methods.  

```python
from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password
```

- **`UserMixin`** provides default methods:  
  - `is_authenticated`
  - `is_active`
  - `is_anonymous`
  - `get_id()`

---

## User Loader  
Define a function to load users by ID:  
```python
@login_manager.user_loader
def load_user(user_id):
    return User(user_id, "admin", "password")  # Replace with actual lookup
```

---

## Login Handling  
Use **Flask-WTF** for the login form:  
```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
```

Process login requests:  
```python
from flask import request, redirect, url_for, render_template
from flask_login import login_user

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User(1, form.username.data, form.password.data)  # Replace with DB lookup
        if user.password == "password":  # Replace with hashed password check
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html', form=form)
```

---

## Logout Handling  
```python
from flask_login import logout_user

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))
```

---

## Protecting Routes  
Restrict access to logged-in users:  
```python
from flask_login import login_required

@app.route('/dashboard')
@login_required
def dashboard():
    return "Welcome to the dashboard!"
```

---

## Checking Authentication  
Use `current_user` to check login state:  
```python
from flask_login import current_user

if current_user.is_authenticated:
    print(f"Logged in as {current_user.username}")
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-login` |
| **Configuration** | Initialize `LoginManager` |
| **User Model** | Use `UserMixin` for authentication methods |
| **User Loader** | `@login_manager.user_loader` loads users from the database |
| **Login Handling** | Use `login_user()` for authentication |
| **Logout Handling** | Use `logout_user()` to clear session |
| **Route Protection** | Use `@login_required` to restrict access |
| **Current User** | Use `current_user` to check login state |
