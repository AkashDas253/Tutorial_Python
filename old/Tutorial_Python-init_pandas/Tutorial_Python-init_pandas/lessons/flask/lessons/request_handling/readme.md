## Request Handling in Flask  

### Overview  
Flask provides the `request` object to handle incoming HTTP requests. It allows access to request data such as form inputs, query parameters, headers, and JSON payloads.

---

### Importing `request`  
```python
from flask import Flask, request

app = Flask(__name__)
```

---

### Accessing Query Parameters (`GET` Requests)  
Query parameters are values passed in the URL after `?`.  

#### Example: `/search?query=Flask`  
```python
@app.route('/search')
def search():
    query = request.args.get('query')  # Retrieves 'query' parameter
    return f"Search results for: {query}"
```
- `request.args.get('query')` extracts the value of `query`.  
- If `query` is missing, it returns `None`.

---

### Handling Form Data (`POST` Requests)  
Form data is sent in `POST` requests.  

#### HTML Form Example:  
```html
<form action="/submit" method="POST">
    <input type="text" name="username">
    <button type="submit">Submit</button>
</form>
```

#### Flask Route:  
```python
@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    return f"Hello, {username}!"
```
- `request.form.get('username')` retrieves form input.  

---

### Handling JSON Data  
For API-based requests, JSON data is sent in the body.  

#### Example JSON Payload:  
```json
{
    "name": "Alice",
    "age": 25
}
```

#### Flask Route:  
```python
from flask import jsonify

@app.route('/json', methods=['POST'])
def json_example():
    data = request.get_json()
    name = data.get('name')
    age = data.get('age')
    return jsonify({"message": f"User {name} is {age} years old"})
```
- `request.get_json()` parses JSON from the request body.  
- `jsonify()` returns a JSON response.  

---

### Accessing Headers  
```python
@app.route('/headers')
def headers():
    user_agent = request.headers.get('User-Agent')
    return f"User Agent: {user_agent}"
```
- `request.headers.get('User-Agent')` retrieves the browser's user agent.

---

### File Upload Handling  
#### HTML Form:  
```html
<form action="/upload" method="POST" enctype="multipart/form-data">
    <input type="file" name="file">
    <button type="submit">Upload</button>
</form>
```

#### Flask Route:  
```python
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(f'uploads/{file.filename}')
    return "File uploaded successfully!"
```
- `request.files['file']` accesses uploaded files.  
- `.save()` stores the file in the `uploads/` directory.  

---

### Checking HTTP Method  
```python
@app.route('/method', methods=['GET', 'POST'])
def check_method():
    if request.method == 'POST':
        return "This is a POST request"
    return "This is a GET request"
```
- `request.method` identifies the request type.  

---

### Accessing Cookies  
```python
@app.route('/setcookie')
def set_cookie():
    response = make_response("Cookie Set")
    response.set_cookie('username', 'Alice')
    return response

@app.route('/getcookie')
def get_cookie():
    username = request.cookies.get('username')
    return f"Username from cookie: {username}"
```
- `response.set_cookie()` sets a cookie.  
- `request.cookies.get('username')` retrieves it.  

---

### Summary  

| Feature | Description |
|---------|------------|
| **Query Parameters** | `request.args.get('param')` |
| **Form Data** | `request.form.get('field')` |
| **JSON Data** | `request.get_json()` |
| **Headers** | `request.headers.get('Header-Name')` |
| **File Uploads** | `request.files['file']` |
| **Request Method** | `request.method` |
| **Cookies** | `request.cookies.get('cookie_name')` |
