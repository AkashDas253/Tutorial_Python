# Environment Variables in Python

## Concept of Environment Variables

* **Definition**: Key–value pairs in the operating system that affect processes and applications.
* **Scope**: System-wide, user-specific, or session-specific.
* **Purpose**:

  * Store configuration (e.g., DB credentials, API keys).
  * Control behavior of programs (e.g., `PATH`, `PYTHONPATH`).
  * Avoid hardcoding sensitive or environment-dependent data.

---

## Usage in Python

* **Access**

  * `os.environ` → dictionary-like object containing all environment variables.
  * `os.getenv(key, default=None)` → safe access with optional default.

* **Modify**

  * `os.environ['VAR'] = 'value'` → set/change variable for current process.
  * Changes affect **only current process and child processes**, not parent OS shell.

* **Delete**

  * `del os.environ['VAR']` → remove a variable from current environment.

---

## Workflow in Applications

* **Loading configuration**

  * Use environment variables instead of storing secrets in code.
  * Example: Database connection, API tokens.

* **Default vs Override**

  * Default values in code.
  * Override using environment variables at runtime (deployment flexibility).

* **Cross-platform differences**

  * Windows, Linux, macOS naming conventions may vary.
  * Case-sensitive on Unix-like OS, case-insensitive on Windows.

---

## Integration with Tools

* **`.env` files**

  * External files for storing environment variables.
  * Load using libraries like `python-dotenv`.

* **Virtual environments**

  * May set `VIRTUAL_ENV`, `PATH` modifications for isolation.

* **Cloud / Containers**

  * Environment variables often injected via Kubernetes, Docker, or CI/CD systems.

---

## Best Practices

* Do not hardcode sensitive data in code.
* Use environment variables for secrets, credentials, and deployment-specific values.
* Use `os.getenv` with defaults to avoid `KeyError`.
* Document required environment variables in project setup.
* Consider `.env` files for local development, but not in production.

---

## Example

```python
import os

# Access
db_user = os.getenv("DB_USER", "default_user")
print("DB_USER:", db_user)

# Set
os.environ["APP_MODE"] = "development"

# Delete
if "OLD_VAR" in os.environ:
    del os.environ["OLD_VAR"]

# Iterate all
for key, value in os.environ.items():
    print(key, "=", value)
```

---
