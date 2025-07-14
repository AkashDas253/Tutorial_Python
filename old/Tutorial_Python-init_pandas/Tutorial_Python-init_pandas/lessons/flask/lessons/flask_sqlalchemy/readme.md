## Flask-SQLAlchemy  

### Overview  
Flask-SQLAlchemy is an extension that integrates SQLAlchemy, a powerful Object Relational Mapper (ORM), with Flask. It simplifies database interactions by mapping Python classes to database tables.

---

## Installation  
Install Flask-SQLAlchemy using pip:  
```sh
pip install flask-sqlalchemy
```

---

## Configuration  
### Connecting to a Database  
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'  # SQLite Database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Disable modification tracking

db = SQLAlchemy(app)
```

| Database | Connection String |
|----------|------------------|
| SQLite | `'sqlite:///data.db'` |
| PostgreSQL | `'postgresql://user:password@localhost/dbname'` |
| MySQL | `'mysql://user:password@localhost/dbname'` |

---

## Defining Models  
### Creating a Model (Table)  
```python
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)

    def __repr__(self):
        return f'<User {self.name}>'
```
- **`db.Integer`** → Integer type  
- **`db.String(100)`** → String with max length 100  
- **`primary_key=True`** → Unique identifier  
- **`nullable=False`** → Cannot be empty  
- **`unique=True`** → No duplicates allowed  

---

## Creating the Database  
Run the following commands to initialize the database:  
```python
with app.app_context():
    db.create_all()  # Creates tables based on models
```

---

## CRUD Operations  

### 1. **Insert Data**
```python
new_user = User(name="Alice", email="alice@example.com")
db.session.add(new_user)
db.session.commit()
```

---

### 2. **Retrieve Data**
```python
users = User.query.all()  # Get all users
user = User.query.filter_by(name="Alice").first()  # Get one user
```

---

### 3. **Update Data**
```python
user = User.query.filter_by(name="Alice").first()
user.email = "newalice@example.com"
db.session.commit()
```

---

### 4. **Delete Data**
```python
user = User.query.filter_by(name="Alice").first()
db.session.delete(user)
db.session.commit()
```

---

## Querying Data  

| Query | Description |
|-------|------------|
| `User.query.all()` | Get all records |
| `User.query.first()` | Get the first record |
| `User.query.get(id)` | Get a record by primary key |
| `User.query.filter_by(name="Alice")` | Filter using `=` |
| `User.query.filter(User.name.like("%A%"))` | Search using `LIKE` |
| `User.query.order_by(User.name.desc())` | Order by descending |

---

## Relationships  
Flask-SQLAlchemy supports **one-to-many**, **many-to-one**, and **many-to-many** relationships.

### One-to-Many Relationship  
```python
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref='posts')
```
- **`db.ForeignKey('user.id')`** → Links Post to User  
- **`db.relationship('User', backref='posts')`** → Enables access via `user.posts`  

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-sqlalchemy` |
| **Configuration** | Set `SQLALCHEMY_DATABASE_URI` |
| **Model Definition** | Use `db.Model` to create tables |
| **CRUD Operations** | `add()`, `query()`, `update()`, `delete()` |
| **Querying Data** | Use `filter_by()`, `filter()`, `order_by()` |
| **Relationships** | `ForeignKey` and `relationship()` for linking tables |
