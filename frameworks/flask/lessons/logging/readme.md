## Logging in Flask  

### Overview  
Flask provides built-in logging support using Python’s `logging` module. Logs help in debugging, tracking errors, and monitoring application activity.

---

## Basic Logging  

### Using Flask’s Default Logger  
```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    app.logger.info("Home page accessed")
    return "Welcome to Flask Logging!"
```
- **`app.logger.info("message")`** logs an info-level message.  
- Flask automatically logs warnings and errors.

---

### Logging Levels  
| Level | Description |
|-------|------------|
| `DEBUG` | Detailed debug information |
| `INFO` | General events like user access |
| `WARNING` | Something unexpected, but the app still works |
| `ERROR` | Major issue causing part of the app to fail |
| `CRITICAL` | Serious failure requiring immediate attention |

---

## Configuring Logging  

### Setting a Custom Logging Format  
```python
import logging

logging.basicConfig(level=logging.DEBUG,  
                    format='%(asctime)s - %(levelname)s - %(message)s',  
                    filename='app.log',  
                    filemode='a')  # Append logs to a file
```
- Logs messages in **`app.log`** with a timestamp.

---

### Logging Different Levels  
```python
@app.route('/test')
def test():
    app.logger.debug("Debug message")
    app.logger.info("Info message")
    app.logger.warning("Warning message")
    app.logger.error("Error message")
    app.logger.critical("Critical error message")
    return "Check logs!"
```
- Each log level provides a different severity.

---

## Logging to a File  

### Example: Writing Logs to a File  
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.ERROR)

app.logger.addHandler(handler)
```
- **`RotatingFileHandler`** ensures logs don’t grow too large.
- **`maxBytes=10000`** → Limits file size to 10 KB.
- **`backupCount=3`** → Keeps the last 3 log files.

---

## Summary  

| Feature | Description |
|---------|------------|
| **Basic Logging** | `app.logger.info("message")` for tracking |
| **Logging Levels** | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| **Custom Format** | `logging.basicConfig(format='...')` for structured logs |
| **File Logging** | `RotatingFileHandler()` prevents oversized log files |
