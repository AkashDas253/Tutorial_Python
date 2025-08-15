## Python’s `logging` Module

The `logging` module in Python provides a flexible framework for emitting log messages from Python programs. It supports multiple logging levels, output formats, and destinations, and allows fine-grained control over how messages are handled.

---

### Overview

* Built-in module (no installation required).
* Supports different severity levels for messages.
* Allows logging to console, files, network sockets, syslog, email, etc.
* Configurable via code or external configuration files.
* Thread-safe and can be used in multi-module applications.

---

### Logging Levels (Severity)

| Level Name | Numeric Value | Purpose                                       |
| ---------- | ------------- | --------------------------------------------- |
| `CRITICAL` | 50            | Very serious errors causing program to abort. |
| `ERROR`    | 40            | Serious problems that require attention.      |
| `WARNING`  | 30            | Indication of potential problems.             |
| `INFO`     | 20            | General information on program execution.     |
| `DEBUG`    | 10            | Detailed information for diagnosing problems. |
| `NOTSET`   | 0             | No level set; used for default handling.      |

---

### Basic Usage

```python
import logging

# Simple configuration
logging.basicConfig(
    level=logging.DEBUG,                      # Minimum log level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Output format
    datefmt="%Y-%m-%d %H:%M:%S",               # Date/time format
    filename="app.log",                        # Log file (optional)
    filemode="a"                               # Append mode
)

logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")
```

---

### Logger Hierarchy

* **Logger** – Main interface for generating log messages (`logging.getLogger(name)`).
* **Handler** – Sends logs to specific destinations (console, file, HTTP, email).
* **Formatter** – Controls log message layout.
* **Filter** – Selectively allows or blocks log records.

---

### Creating Custom Loggers

```python
logger = logging.getLogger("my_app")  
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("my_app.log")
file_handler.setLevel(logging.ERROR)

# Formatter
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger.info("This goes to console")
logger.error("This goes to both console and file")
```

---

### Configuration Methods

#### 1. **Basic Configuration**

* Quick setup with `logging.basicConfig(...)`.

#### 2. **DictConfig** (from `logging.config`)

* Load configuration from a Python dictionary.

#### 3. **FileConfig**

* Load configuration from `.ini` file format.

---

### Log Record Attributes

Some common format variables:

* `%(asctime)s` – Timestamp.
* `%(name)s` – Logger name.
* `%(levelname)s` – Severity level name.
* `%(message)s` – Log message.
* `%(filename)s` – File name where log was issued.
* `%(lineno)d` – Line number of logging call.
* `%(threadName)s` – Thread name.
* `%(process)d` – Process ID.

---

### Handlers

| Handler Class              | Purpose                               |
| -------------------------- | ------------------------------------- |
| `StreamHandler`            | Logs to console or any stream.        |
| `FileHandler`              | Logs to a file.                       |
| `RotatingFileHandler`      | Rotates logs based on size.           |
| `TimedRotatingFileHandler` | Rotates logs based on time.           |
| `SMTPHandler`              | Sends logs via email.                 |
| `HTTPHandler`              | Sends logs via HTTP POST/GET.         |
| `SysLogHandler`            | Logs to Unix syslog or remote syslog. |
| `NullHandler`              | Ignores log messages.                 |

---

### Filters

* Used to allow/deny log records dynamically.
* Example:

  ```python
  class InfoFilter(logging.Filter):
      def filter(self, record):
          return record.levelno == logging.INFO

  logger.addFilter(InfoFilter())
  ```

---

### Usage Scenarios

* Debugging and troubleshooting during development.
* Recording application events in production.
* Tracking errors and warnings over time.
* Integrating with monitoring systems (via HTTP, email, syslog).

---
