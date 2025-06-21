## Database Integration in Flask  

### Overview  
Flask supports multiple databases using **Flask-SQLAlchemy** and **Flask-Migrate** for integration and migrations.

---

## Installation  
Install Flask-SQLAlchemy and Flask-Migrate:  
```sh
pip install flask-sqlalchemy flask-migrate
```

---

## Configuring Database  
Set up the database URI in `config`:  
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Change for PostgreSQL/MySQL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
```

---

## Defining Models  
```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
```

---

## Creating the Database  
```sh
flask db init        # Initialize migration directory
flask db migrate -m "Initial migration"
flask db upgrade    # Apply migrations
```

---

## CRUD Operations  

| Operation | Method |
|-----------|--------|
| **Create** | `db.session.add(obj)`, `db.session.commit()` |
| **Read** | `User.query.all()`, `User.query.get(id)` |
| **Update** | Modify object, then `db.session.commit()` |
| **Delete** | `db.session.delete(obj)`, `db.session.commit()` |

### Example  
```python
# Create a new user
new_user = User(username="john_doe", email="john@example.com")
db.session.add(new_user)
db.session.commit()

# Read users
users = User.query.all()

# Update user
user = User.query.get(1)
user.email = "new_email@example.com"
db.session.commit()

# Delete user
db.session.delete(user)
db.session.commit()
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-sqlalchemy flask-migrate` |
| **Configuration** | Define `SQLALCHEMY_DATABASE_URI` |
| **Defining Models** | Create a class with `db.Model` |
| **Migrations** | Use `flask db migrate` and `flask db upgrade` |
| **CRUD Operations** | Perform `Create, Read, Update, Delete` using `db.session` |
