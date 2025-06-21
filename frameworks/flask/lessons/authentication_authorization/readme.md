## Authentication & Authorization in Flask  

### Overview  
Flask provides authentication and authorization using **Flask-Login** for user sessions and **Flask-WTF** for secure form handling.

---

## Installation  
```sh
pip install flask-login flask-wtf flask-bcrypt
```

---

## Configuration  
Set up Flask-Login and bcrypt:  
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SECRET_KEY'] = 'your_secret_key'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
bcrypt = Bcrypt(app)
login_manager.login_view = "login"
```

---

## User Model  
```python
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
```

---

## Loading Users  
```python
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
```

---

## Registration Form  
```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Register')
```

---

## Register Route  
```python
from flask import render_template, redirect, url_for, flash
from flask_login import login_user

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)
```

---

## Login Form  
```python
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
```

---

## Login Route  
```python
from flask_login import login_required, logout_user, current_user

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Login failed. Check your email and password.', 'danger')
    return render_template('login.html', form=form)
```

---

## Logout Route  
```python
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
```

---

## Protected Route (Authorization)  
```python
@app.route('/dashboard')
@login_required
def dashboard():
    return f"Welcome, {current_user.username}!"
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-login flask-wtf flask-bcrypt` |
| **User Model** | Define `User` with `UserMixin` |
| **User Loading** | Implement `@login_manager.user_loader` |
| **Registration** | Use `bcrypt` for password hashing |
| **Login** | Verify passwords and authenticate users |
| **Authorization** | Protect routes with `@login_required` |
| **Logout** | Use `logout_user()` to end session |
