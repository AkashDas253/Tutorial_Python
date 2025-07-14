## REST API Development in Flask  

### Overview  
Flask provides REST API capabilities using **Flask-RESTful**, allowing easy route handling, request parsing, and response formatting.

---

## Installation  
```sh
pip install flask flask-restful
```

---

## Setting Up Flask-RESTful  
```python
from flask import Flask
from flask_restful import Api

app = Flask(__name__)
api = Api(app)
```

---

## Creating a Resource  
```python
from flask_restful import Resource

class HelloWorld(Resource):
    def get(self):
        return {"message": "Hello, World!"}

api.add_resource(HelloWorld, '/')
```

---

## Running the API  
```python
if __name__ == '__main__':
    app.run(debug=True)
```

---

## Handling CRUD Operations  
```python
class Item(Resource):
    items = []

    def get(self, name):
        for item in self.items:
            if item["name"] == name:
                return item, 200
        return {"message": "Item not found"}, 404

    def post(self, name):
        item = {"name": name}
        self.items.append(item)
        return item, 201

    def delete(self, name):
        self.items = [item for item in self.items if item["name"] != name]
        return {"message": "Item deleted"}, 200

api.add_resource(Item, '/item/<string:name>')
```

---

## Request Parsing  
```python
from flask_restful import reqparse

parser = reqparse.RequestParser()
parser.add_argument("price", type=float, required=True, help="Price cannot be blank")

class Item(Resource):
    def post(self, name):
        args = parser.parse_args()
        item = {"name": name, "price": args["price"]}
        self.items.append(item)
        return item, 201
```

---

## Using JSON Responses  
```python
from flask import jsonify

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404
```

---

## Authentication for API (Token-Based)  
```python
from flask import request

API_KEY = "secret123"

def authenticate(func):
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization")
        if token != API_KEY:
            return {"message": "Unauthorized"}, 401
        return func(*args, **kwargs)
    return wrapper

class SecureResource(Resource):
    @authenticate
    def get(self):
        return {"message": "Authenticated"}
    
api.add_resource(SecureResource, '/secure')
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Installation** | `pip install flask flask-restful` |
| **API Setup** | Initialize `Flask` and `Api` |
| **Resource Handling** | Define classes extending `Resource` |
| **CRUD Operations** | Implement `get()`, `post()`, `delete()` |
| **Request Parsing** | Use `reqparse.RequestParser()` |
| **Authentication** | Implement token-based authentication |
