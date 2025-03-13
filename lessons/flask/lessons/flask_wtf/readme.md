## Flask-WTF  

### Overview  
Flask-WTF is an extension that integrates **WTForms** with Flask, simplifying form handling, validation, and CSRF protection.

---

## Installation  
Install Flask-WTF using pip:  
```sh
pip install flask-wtf
```

---

## Configuration  
Enable CSRF protection in the Flask app:  
```python
from flask import Flask
from flask_wtf import CSRFProtect

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for CSRF protection
csrf = CSRFProtect(app)
```

---

## Creating Forms  
Define forms using `FlaskForm` with field types and validators.  

```python
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Login')
```

| Field Type | Description |
|------------|------------|
| `StringField` | Input text field |
| `PasswordField` | Hidden password input |
| `SubmitField` | Submit button |
| `BooleanField` | Checkbox field |
| `IntegerField` | Numeric input |
| `TextAreaField` | Multi-line input |

| Validator | Description |
|-----------|------------|
| `DataRequired()` | Ensures the field is not empty |
| `Email()` | Validates email format |
| `Length(min, max)` | Restricts input length |
| `EqualTo(field)` | Matches another field |

---

## Rendering Forms in Templates  
In `templates/login.html`:  
```html
<form method="POST">
    {{ form.hidden_tag() }}  <!-- CSRF Token -->
    {{ form.email.label }} {{ form.email() }}
    {{ form.password.label }} {{ form.password() }}
    {{ form.submit() }}
</form>
```

---

## Handling Forms in Routes  
```python
from flask import render_template, request

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        return f'Logged in as {email}'
    return render_template('login.html', form=form)
```

---

## CSRF Protection  
Flask-WTF automatically protects against **Cross-Site Request Forgery (CSRF)**.  
Use `{{ form.hidden_tag() }}` in templates to include CSRF tokens.

---

## File Uploads  
```python
from flask_wtf.file import FileField, FileAllowed

class UploadForm(FlaskForm):
    file = FileField('Upload File', validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField('Upload')
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-wtf` |
| **Configuration** | Set `SECRET_KEY` for CSRF |
| **Form Definition** | Use `FlaskForm` with fields & validators |
| **Rendering Forms** | Use `{{ form.hidden_tag() }}` in templates |
| **Handling Forms** | `form.validate_on_submit()` to process input |
| **CSRF Protection** | Enabled by default |
| **File Uploads** | Use `FileField` with `FileAllowed()` |
