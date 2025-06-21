## Error Handling in Flask  

### Overview  
Flask provides built-in error handling mechanisms to manage HTTP errors and unexpected exceptions. You can define custom error pages, use exception handling, and log errors for debugging.

---

## Handling HTTP Errors  

### Using `@app.errorhandler()`  
You can define custom error pages for specific HTTP status codes.

#### **Example: Handling 404 Errors (Page Not Found)**
```python
from flask import Flask, render_template

app = Flask(__name__)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404
```
- If a user accesses a non-existent page, **404.html** is displayed.

#### **templates/404.html**
```html
<!DOCTYPE html>
<html>
<head>
    <title>Page Not Found</title>
</head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>Sorry, the page you requested does not exist.</p>
</body>
</html>
```

---

### Handling Multiple Errors  
```python
@app.errorhandler(403)
def forbidden(error):
    return "403 Forbidden", 403

@app.errorhandler(500)
def internal_server_error(error):
    return "500 Internal Server Error", 500
```
- **403 Forbidden:** Access to a resource is denied.  
- **500 Internal Server Error:** Unexpected server-side issue.

---

## Handling Exceptions  

### Using `try-except`  
Handle unexpected errors in routes.

```python
@app.route('/divide/<int:num>')
def divide(num):
    try:
        result = 10 / num
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Cannot divide by zero!", 400
```
- Prevents division by zero errors.

---

### Logging Errors  
Use Flask’s built-in logger to record errors.

```python
import logging

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Server Error: {error}")
    return "500 Internal Server Error", 500
```
- Stores errors in the log for debugging.

---

## Summary  

| Feature | Description |
|---------|------------|
| **Custom Error Pages** | Use `@app.errorhandler(status_code)` |
| **404 Not Found** | Handle missing pages with a custom template |
| **403 Forbidden** | Restrict access to unauthorized users |
| **500 Internal Server Error** | Handle unexpected crashes |
| **Try-Except** | Catch exceptions in routes |
| **Logging** | `app.logger.error("message")` for debugging |
