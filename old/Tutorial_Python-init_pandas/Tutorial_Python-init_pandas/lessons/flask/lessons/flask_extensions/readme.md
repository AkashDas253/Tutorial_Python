## Flask Extensions  

### Overview  
Flask extensions add extra functionality like database support, authentication, form handling, and more. Extensions are installed via `pip` and integrated into the Flask app.

---

## Common Flask Extensions  

| Extension | Purpose |
|-----------|---------|
| **Flask-SQLAlchemy** | Database ORM for handling SQL databases |
| **Flask-Migrate** | Database migrations using Alembic |
| **Flask-WTF** | Form handling with CSRF protection |
| **Flask-Login** | User authentication and session management |
| **Flask-Mail** | Email sending support |
| **Flask-Caching** | Caching support for performance optimization |
| **Flask-RESTful** | Building REST APIs |
| **Flask-JWT-Extended** | JWT-based authentication |

---

## Installing Flask Extensions  
Install any extension using `pip`:

```sh
pip install flask-sqlalchemy flask-migrate flask-wtf flask-login flask-mail flask-caching flask-restful flask-jwt-extended
```

---

## Using Flask Extensions  

### 1. **Flask-SQLAlchemy (Database ORM)**
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))

db.create_all()
```

---

### 2. **Flask-Migrate (Database Migrations)**
```sh
pip install flask-migrate
```
```python
from flask_migrate import Migrate

migrate = Migrate(app, db)
```
```sh
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

---

### 3. **Flask-WTF (Forms Handling)**
```python
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class NameForm(FlaskForm):
    name = StringField('Name', validators=[DataRequired()])
    submit = SubmitField('Submit')
```

---

### 4. **Flask-Login (User Authentication)**
```python
from flask_login import LoginManager

login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect if not logged in
```

---

### 5. **Flask-Mail (Email Support)**
```python
from flask_mail import Mail

app.config['MAIL_SERVER'] = 'smtp.example.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@example.com'
app.config['MAIL_PASSWORD'] = 'your-password'

mail = Mail(app)
```

---

### 6. **Flask-Caching (Performance Optimization)**
```python
from flask_caching import Cache

app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

@app.route('/')
@cache.cached(timeout=60)
def index():
    return "Cached for 60 seconds"
```

---

### 7. **Flask-RESTful (API Development)**
```python
from flask_restful import Resource, Api

api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {'message': 'Hello, World!'}

api.add_resource(HelloWorld, '/')
```

---

### 8. **Flask-JWT-Extended (JWT Authentication)**
```python
from flask_jwt_extended import JWTManager

app.config['JWT_SECRET_KEY'] = 'your-secret-key'
jwt = JWTManager(app)
```

---

## Summary  

| Extension | Functionality |
|-----------|--------------|
| **Flask-SQLAlchemy** | ORM for database interactions |
| **Flask-Migrate** | Database migrations |
| **Flask-WTF** | Form validation & CSRF protection |
| **Flask-Login** | User authentication & session management |
| **Flask-Mail** | Sending emails |
| **Flask-Caching** | Performance optimization |
| **Flask-RESTful** | Building REST APIs |
| **Flask-JWT-Extended** | JWT-based authentication |
