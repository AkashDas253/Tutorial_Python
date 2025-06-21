## Response Handling in Flask  

### Overview  
Flask provides various ways to send responses, including plain text, JSON, HTML templates, redirects, and custom headers. The `Response` object allows fine-grained control over the response.

---

### Basic String Response  
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Flask!"
```
- The function returns a simple string as an HTTP response.  

---

### Returning HTML Content  
```python
@app.route('/html')
def html_response():
    return "<h1>Hello, Flask!</h1>"
```
- The response contains raw HTML content.  

---

### Using `make_response()`  
The `make_response()` function creates a custom response.  
```python
from flask import make_response

@app.route('/custom')
def custom_response():
    response = make_response("Custom Response", 200)
    response.headers["Content-Type"] = "text/plain"
    return response
```
- Allows setting the status code and headers.  

---

### Returning JSON Responses  
Use `jsonify()` to return structured JSON data.  
```python
from flask import jsonify

@app.route('/json')
def json_response():
    data = {"message": "Hello, Flask!", "status": "success"}
    return jsonify(data)
```
- Converts a dictionary into a JSON response.  

#### Custom Status Code with `jsonify()`  
```python
@app.route('/error')
def error_response():
    return jsonify({"error": "Invalid request"}), 400
```
- Returns JSON with a 400 (Bad Request) status.  

---

### Redirecting Users  
Use `redirect()` to send users to another URL.  
```python
from flask import redirect, url_for

@app.route('/old')
def old_route():
    return redirect(url_for('home'))
```
- Redirects `/old` to `/`.  

---

### Setting Custom Headers  
```python
@app.route('/headers')
def custom_headers():
    response = make_response("Hello with custom headers!")
    response.headers['X-Custom-Header'] = "FlaskApp"
    return response
```
- `X-Custom-Header` is added to the response.  

---

### Sending Files as Response  
```python
from flask import send_file

@app.route('/download')
def download_file():
    return send_file("static/sample.pdf", as_attachment=True)
```
- Allows file downloads with `send_file()`.  

---

### Setting Cookies  
```python
@app.route('/setcookie')
def set_cookie():
    response = make_response("Cookie set!")
    response.set_cookie('username', 'Alice', max_age=3600)  # 1-hour expiry
    return response

@app.route('/getcookie')
def get_cookie():
    username = request.cookies.get('username')
    return f"Username from cookie: {username}"
```
- `set_cookie()` sets a cookie.  
- `request.cookies.get()` retrieves it.  

---

### Deleting Cookies  
```python
@app.route('/deletecookie')
def delete_cookie():
    response = make_response("Cookie deleted!")
    response.delete_cookie('username')
    return response
```
- Removes the `username` cookie.  

---

### Summary  

| Feature | Description |
|---------|------------|
| **Basic Response** | Returns plain text or HTML |
| **Custom Response** | Uses `make_response()` for headers and status codes |
| **JSON Response** | Uses `jsonify()` for structured data |
| **Redirect** | Uses `redirect(url_for())` to change routes |
| **Custom Headers** | Adds headers using `response.headers[]` |
| **File Response** | Sends files using `send_file()` |
| **Cookies** | `set_cookie()` and `request.cookies.get()` for handling cookies |
