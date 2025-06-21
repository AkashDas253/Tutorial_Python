## Forms Handling in Flask  

### Overview  
Flask handles forms using **Flask-WTF**, an extension that integrates **WTForms** with Flask, providing form validation and CSRF protection.

---

## Installation  
Install Flask-WTF:  
```sh
pip install flask-wtf
```

---

## Configuration  
Set up **CSRF protection** in `config`:  
```python
from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
```

---

## Creating a Form  
Define a form using `FlaskForm`:  
```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')
```

---

## Rendering Forms in Templates  
Use **Jinja2** to display forms in `templates/login.html`:  
```html
<form method="POST">
    {{ form.hidden_tag() }}
    {{ form.email.label }} {{ form.email }}
    {{ form.password.label }} {{ form.password }}
    {{ form.submit }}
</form>
```

---

## Handling Form Submission  
```python
from flask import render_template, request

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        return f"Welcome, {form.email.data}"
    return render_template('login.html', form=form)
```

---

## Validation & Error Handling  
```html
{% for field, errors in form.errors.items() %}
    {% for error in errors %}
        <p style="color:red;">{{ field }}: {{ error }}</p>
    {% endfor %}
{% endfor %}
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-wtf` |
| **CSRF Protection** | Set `SECRET_KEY` |
| **Defining Forms** | Use `FlaskForm` and `wtforms` fields |
| **Rendering Forms** | Use `form.hidden_tag()` and fields in Jinja2 |
| **Validation** | Use `form.validate_on_submit()` |
| **Error Handling** | Display `form.errors` in templates |
