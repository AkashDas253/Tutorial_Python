# `requests` Module (Python)

## Introduction

* `requests` is a high-level HTTP library for Python used to send HTTP/1.1 requests.
* Simplifies making web requests without manually handling socket connections or building HTTP headers.
* Supports a wide range of HTTP methods and advanced features.

---

## Core Concepts

### HTTP Methods

* `GET` — Retrieve data from the server.
* `POST` — Send data to the server.
* `PUT` — Update a resource.
* `DELETE` — Remove a resource.
* `PATCH` — Partial update of a resource.
* `HEAD` — Retrieve headers only.
* `OPTIONS` — Retrieve supported HTTP methods.

---

## Key Functions

| Function                                             | Purpose                  |
| ---------------------------------------------------- | ------------------------ |
| `requests.get(url, **kwargs)`                        | Send a GET request.      |
| `requests.post(url, data=None, json=None, **kwargs)` | Send a POST request.     |
| `requests.put(url, data=None, **kwargs)`             | Send a PUT request.      |
| `requests.delete(url, **kwargs)`                     | Send a DELETE request.   |
| `requests.patch(url, data=None, **kwargs)`           | Send a PATCH request.    |
| `requests.head(url, **kwargs)`                       | Send a HEAD request.     |
| `requests.options(url, **kwargs)`                    | Send an OPTIONS request. |

---

## Common Parameters (for all request methods)

* `params` — Dictionary or bytes to send in query string.
* `data` — Dictionary, bytes, or file-like object to send in the body.
* `json` — JSON data to send in the body.
* `headers` — Dictionary of HTTP headers.
* `cookies` — Dictionary or CookieJar of cookies.
* `auth` — Tuple for basic auth `(username, password)`.
* `timeout` — Time (seconds) before request times out.
* `allow_redirects` — Boolean to follow redirects (default `True` for most).
* `proxies` — Dictionary mapping protocol to proxy URL.
* `verify` — Boolean or path to CA bundle for SSL verification.
* `stream` — Whether to stream the response content.
* `files` — Dictionary for file uploads (`{'file': open('filename', 'rb')}`).

---

## Response Object (`requests.Response`)

* `.status_code` — HTTP status code.
* `.headers` — Response headers.
* `.text` — Response body as string.
* `.content` — Response body as bytes.
* `.json()` — Parse JSON response.
* `.cookies` — Cookies from the response.
* `.url` — Final URL after redirects.
* `.history` — List of `Response` objects from redirects.
* `.elapsed` — Time taken for request.

---

## Session Objects (`requests.Session`)

* Maintain cookies across requests.
* Persist certain parameters across requests.
* More efficient for multiple requests to the same host.

**Key Methods:**

* `.get()`, `.post()`, etc. — Same as top-level functions but session-based.
* `.headers.update()` — Set default headers.
* `.cookies` — Manage session cookies.

---

## Advanced Features

* **File Uploads** — Use `files` parameter.
* **Streaming Downloads** — Use `stream=True` and iterate over `Response.iter_content()`.
* **Custom Authentication** — Pass `auth` with HTTPBasicAuth, OAuth, etc.
* **SSL Verification** — `verify=False` for skipping SSL (not recommended).
* **Timeouts** — Set per-request or per-session.
* **Proxy Support** — `proxies={'http': 'http://proxy.com', 'https': 'https://proxy.com'}`.
* **Hooks** — Callbacks triggered on events like receiving a response.

---

## Exceptions (in `requests.exceptions`)

* `RequestException` — Base class for all exceptions.
* `HTTPError` — Raised for HTTP errors.
* `ConnectionError` — Network-related errors.
* `Timeout` — Request timed out.
* `TooManyRedirects` — Exceeded max redirects.
* `URLRequired` — Invalid or missing URL.

---

## Syntax Examples

```python
import requests

# GET request with query parameters
response = requests.get(
    'https://api.example.com/data',
    params={'key': 'value'},
    headers={'Accept': 'application/json'},
    timeout=5
)
print(response.status_code, response.json())

# POST request with JSON body
response = requests.post(
    'https://api.example.com/create',
    json={'name': 'Alice', 'age': 25}
)

# Session example
session = requests.Session()
session.headers.update({'User-Agent': 'my-app'})
response = session.get('https://example.com')
print(response.text)
```

---
