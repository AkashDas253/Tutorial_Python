## Testing and Debugging in Flask  

### Overview  
Flask provides built-in tools for testing and debugging applications. This includes:  

- **Debug Mode**: Enables live reloading and error tracking  
- **Unit Testing**: Using `unittest` and `pytest` for automated tests  
- **Logging**: Tracking issues with Flask's built-in logging system  

---

## Enabling Debug Mode  
Run Flask with **debug mode** to auto-restart on code changes and display error messages.  

```sh
export FLASK_ENV=development  # Linux/Mac
set FLASK_ENV=development     # Windows
flask run
```
OR  
```python
app.run(debug=True)
```

---

## Debugging with Flask Debug Toolbar  
Install:  
```sh
pip install flask-debugtoolbar
```
Usage:  
```python
from flask import Flask
from flask_debugtoolbar import DebugToolbarExtension

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['DEBUG_TB_INTERCEPT_REDIRECTS'] = False  # Avoid redirect interception

toolbar = DebugToolbarExtension(app)
```

---

## Logging Errors  
```python
import logging

logging.basicConfig(filename='app.log', level=logging.DEBUG)

@app.route('/error')
def error():
    try:
        1 / 0  # Intentional error
    except ZeroDivisionError as e:
        app.logger.error(f"Error occurred: {e}")
        return "An error occurred!", 500
```

---

## Unit Testing with `unittest`  
```python
import unittest
from app import app

class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
```
Run tests:  
```sh
python -m unittest discover
```

---

## Testing with `pytest`  
Install:  
```sh
pip install pytest
```
Usage:  
```python
import pytest
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
```
Run tests:  
```sh
pytest
```

---

## Flask Test Client (Simulating Requests)  
```python
def test_post_request(client):
    response = client.post('/submit', data={'name': 'Alice'})
    assert response.status_code == 200
    assert b"Alice" in response.data
```

---

## Summary  

| Feature | Description |
|---------|------------|
| **Debug Mode** | `FLASK_ENV=development` for live reloading |
| **Logging** | `app.logger.error("Message")` to track issues |
| **Debug Toolbar** | `flask-debugtoolbar` for interactive debugging |
| **Unit Testing** | `unittest` framework for automated tests |
| **Pytest** | `pytest` for simple and flexible testing |
