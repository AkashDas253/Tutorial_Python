## Allowed Host in Django

### Purpose

* Security setting that **restricts which host/domain names the Django site can serve**.
* Prevents **HTTP Host header attacks**.
* Used when **`DEBUG=False`** (production mode).

---

### Location

* Defined in **`settings.py`** as `ALLOWED_HOSTS`.

---

### Syntax

```python
ALLOWED_HOSTS = [
    'example.com',     # Exact domain
    '.example.com',    # Any subdomain of example.com
    'localhost',       # Local development
    '127.0.0.1',       # Local IP
    '[::1]',           # IPv6 loopback
    '*'                # (Not recommended) Allow all hosts
]
```

---

### Behavior

* When a request comes in, Django checks the **`Host` header** in the HTTP request.
* If the value is not listed in **`ALLOWED_HOSTS`**, Django raises **`SuspiciousOperation`** error.

---

### Best Practices

* Always use **exact domain names** or **subdomain patterns** in production.
* Avoid using `'*'` in production — only for quick testing.
* Use environment variables to set `ALLOWED_HOSTS` dynamically.

---
