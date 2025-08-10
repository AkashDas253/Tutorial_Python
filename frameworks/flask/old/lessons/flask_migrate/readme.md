## Flask-Migrate  

### Overview  
Flask-Migrate is an extension that integrates **Alembic** with **Flask-SQLAlchemy** to handle database migrations.

---

## Installation  
Install Flask-Migrate and Alembic:  
```sh
pip install flask-migrate
```

---

## Configuration  
Initialize Flask-Migrate with Flask and SQLAlchemy:  
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # Change for PostgreSQL/MySQL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
```

---

## Initializing Migrations  
Run these commands in the terminal:  
```sh
flask db init  # Initialize Alembic migration directory
flask db migrate -m "Initial migration"  # Generate migration script
flask db upgrade  # Apply migration to the database
```

---

## Creating a Model  
```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
```

After defining a model, generate and apply a migration:  
```sh
flask db migrate -m "Add User model"
flask db upgrade
```

---

## Managing Database Versions  

| Command | Description |
|---------|------------|
| `flask db init` | Initializes migration setup |
| `flask db migrate -m "message"` | Generates a migration script |
| `flask db upgrade` | Applies the migration |
| `flask db downgrade` | Reverts the last migration |
| `flask db history` | Shows migration history |
| `flask db current` | Displays the current migration version |

---

## Rolling Back a Migration  
To undo the last migration:  
```sh
flask db downgrade
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-migrate` |
| **Configuration** | Initialize `Migrate(app, db)` |
| **Creating Migrations** | Use `flask db migrate` and `flask db upgrade` |
| **Rolling Back** | Use `flask db downgrade` |
| **Database Versioning** | Track changes using `flask db history` |
