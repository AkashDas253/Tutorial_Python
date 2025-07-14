# Internet related modules

## Python JSON

### JSON in Python

- JSON is a syntax for storing and exchanging data.
- JSON is text, written with JavaScript object notation.
- Python has a built-in package called `json` to work with JSON data.

### Import the JSON Module

#### Syntax
```python
import json
```

### Parse JSON - Convert from JSON to Python

- Use the `json.loads()` method to parse a JSON string into a Python dictionary.

#### Syntax
```python
import json

# some JSON:
x = '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])
```

### Convert from Python to JSON

- Use the `json.dumps()` method to convert a Python object into a JSON string.

#### Syntax
```python
import json

# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)
```

### Convert Various Python Objects to JSON Strings

- You can convert Python objects of the following types into JSON strings: `dict`, `list`, `tuple`, `string`, `int`, `float`, `True`, `False`, `None`.

#### Syntax
```python
import json

print(json.dumps({"name": "John", "age": 30}))
print(json.dumps(["apple", "bananas"]))
print(json.dumps(("apple", "bananas")))
print(json.dumps("hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))
```

### Python to JSON Conversion Table

- Python objects are converted into the JSON (JavaScript) equivalent:

| Python | JSON    |
|--------|---------|
| dict   | Object  |
| list   | Array   |
| tuple  | Array   |
| str    | String  |
| int    | Number  |
| float  | Number  |
| True   | true    |
| False  | false   |
| None   | null    |

### Convert a Python Object Containing All Legal Data Types

#### Syntax
```python
import json

x = {
  "name": "John",
  "age": 30,
  "married": True,
  "divorced": False,
  "children": ("Ann","Billy"),
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

print(json.dumps(x))
```

### Format the Result

- Use the `indent` parameter to define the number of indents.

#### Syntax
```python
json.dumps(x, indent=4)
```

- Use the `separators` parameter to change the default separator.

#### Syntax
```python
json.dumps(x, indent=4, separators=(". ", " = "))
```

### Order the Result

- Use the `sort_keys` parameter to specify if the result should be sorted or not.

#### Syntax
```python
json.dumps(x, indent=4, sort_keys=True)
```

## HTTP Methods

### Install:

```python
pip install requests
```

### Syntax:

```python
requests.methodname(params)
```

### HTTP Method

- `delete(url, args)` - Sends a DELETE request to the specified URL.
  - `url`: The URL to send the request to.
  - `args`: Additional arguments to pass with the request.

- `get(url, params, args)` - Sends a GET request to the specified URL.
  - `url`: The URL to send the request to.
  - `params`: The parameters to include in the query string.
  - `args`: Additional arguments to pass with the request.

- `head(url, args)` - Sends a HEAD request to the specified URL.
  - `url`: The URL to send the request to.
  - `args`: Additional arguments to pass with the request.

- `patch(url, data, args)` - Sends a PATCH request to the specified URL.
  - `url`: The URL to send the request to.
  - `data`: The data to include in the body of the request.
  - `args`: Additional arguments to pass with the request.

- `post(url, data, json, args)` - Sends a POST request to the specified URL.
  - `url`: The URL to send the request to.
  - `data`: The data to include in the body of the request.
  - `json`: The JSON data to include in the body of the request.
  - `args`: Additional arguments to pass with the request.

- `put(url, data, args)` - Sends a PUT request to the specified URL.
  - `url`: The URL to send the request to.
  - `data`: The data to include in the body of the request.
  - `args`: Additional arguments to pass with the request.

- `request(method, url, args)` - Sends a request of the specified method to the specified URL.
  - `method`: The HTTP method to use (e.g., 'GET', 'POST').
  - `url`: The URL to send the request to.
  - `args`: Additional arguments to pass with the request.