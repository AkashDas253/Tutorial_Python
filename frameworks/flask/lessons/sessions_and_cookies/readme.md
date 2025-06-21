## Sessions and Cookies in Flask  

### Overview  
- **Cookies** store small pieces of data on the client’s browser.  
- **Sessions** store user-specific data across multiple requests using cookies.  
- Flask **sessions** use **signed cookies**, ensuring security with a secret key.

---

## Cookies  

### Setting a Cookie  
```python
from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/setcookie')
def set_cookie():
    response = make_response("Cookie Set!")
    response.set_cookie('username', 'Alice', max_age=3600)  # 1-hour expiry
    return response
```
- `set_cookie(name, value, max_age)` sets a cookie in the response.  

---

### Getting a Cookie  
```python
@app.route('/getcookie')
def get_cookie():
    username = request.cookies.get('username')
    return f"Stored username: {username}"
```
- `request.cookies.get(name)` retrieves a cookie.  

---

### Deleting a Cookie  
```python
@app.route('/deletecookie')
def delete_cookie():
    response = make_response("Cookie Deleted!")
    response.delete_cookie('username')
    return response
```
- `delete_cookie(name)` removes the cookie.  

---

## Sessions  

### Enabling Sessions  
- Sessions require a **secret key** to prevent tampering.  

```python
from flask import Flask, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production
```

---

### Setting a Session Variable  
```python
@app.route('/set_session')
def set_session():
    session['user'] = 'Alice'
    return "Session Set!"
```
- `session['key'] = value` stores data in the session.  

---

### Getting a Session Variable  
```python
@app.route('/get_session')
def get_session():
    user = session.get('user', 'Guest')
    return f"Logged in as: {user}"
```
- `session.get('key', default)` retrieves a session value.  

---

### Removing a Session Variable  
```python
@app.route('/remove_session')
def remove_session():
    session.pop('user', None)
    return "Session Removed!"
```
- `session.pop('key', None)` deletes a session key.  

---

### Clearing All Session Data  
```python
@app.route('/clear_session')
def clear_session():
    session.clear()
    return "All Sessions Cleared!"
```
- `session.clear()` removes all session data.  

---

## Summary  

| Feature | Description |
|---------|------------|
| **Cookies** | Stored on the client side, used for small data storage |
| **Set Cookie** | `set_cookie('key', 'value', max_age)` |
| **Get Cookie** | `request.cookies.get('key')` |
| **Delete Cookie** | `delete_cookie('key')` |
| **Sessions** | Stored on the server, signed with a secret key |
| **Set Session** | `session['key'] = value` |
| **Get Session** | `session.get('key', default)` |
| **Remove Session** | `session.pop('key', None)` |
| **Clear All Sessions** | `session.clear()` |
