## Flask-RESTful  

### Overview  
Flask-RESTful is an extension for building **REST APIs** in Flask. It simplifies request parsing, response formatting, and route management.

---

## Installation  
Install Flask-RESTful using pip:  
```sh
pip install flask-restful
```

---

## Configuration  
Initialize Flask-RESTful with Flask:  
```python
from flask import Flask
from flask_restful import Api

app = Flask(__name__)
api = Api(app)
```

---

## Creating a Resource  
Define a resource by subclassing `Resource`:  
```python
from flask_restful import Resource

class HelloWorld(Resource):
    def get(self):
        return {"message": "Hello, World!"}
```

---

## Adding Routes  
Register resources with API routes:  
```python
api.add_resource(HelloWorld, '/')
```

---

## Handling HTTP Methods  
```python
class User(Resource):
    def get(self, user_id):
        return {"user_id": user_id, "name": "John Doe"}

    def post(self, user_id):
        return {"message": f"User {user_id} created"}, 201

api.add_resource(User, '/user/<int:user_id>')
```

| Method | Description |
|--------|------------|
| `get()` | Retrieve data |
| `post()` | Create new data |
| `put()` | Update existing data |
| `delete()` | Remove data |

---

## Request Parsing  
Use `reqparse` to handle input validation:  
```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument('name', type=str, required=True, help="Name cannot be blank")

class User(Resource):
    def post(self):
        args = parser.parse_args()
        return {"message": f"User {args['name']} created"}, 201
```

---

## Error Handling  
Customize error responses:  
```python
api.handle_error = lambda e: ({"error": str(e)}, 400)
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask-restful` |
| **Configuration** | Initialize `Api(app)` |
| **Defining Resources** | Subclass `Resource` and define methods |
| **Routing** | Use `api.add_resource()` |
| **Request Handling** | Use `reqparse` for input validation |
| **Error Handling** | Customize error responses |
