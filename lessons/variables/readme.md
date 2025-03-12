## Variables in Python  

### Definition  
- A variable is a named reference to a value stored in memory.  
- Python variables are dynamically typed, meaning they do not require explicit type declarations.  

### Declaration and Assignment  
- Variables are assigned using `=`:  
  ```python
  x = 10
  name = "Alice"
  pi = 3.14
  ```  
- Multiple assignments:  
  ```python
  a, b, c = 1, 2, 3
  x = y = z = 100  # All three variables reference the same value
  ```  

### Naming Rules  
- Must start with a letter (A-Z, a-z) or an underscore `_`.  
- Can contain letters, digits (0-9), and underscores.  
- Cannot be a Python keyword (e.g., `if`, `for`, `return`).  
- Case-sensitive (`var`, `Var`, and `VAR` are different).  

### Variable Scope  
| Scope Type   | Description |
|-------------|------------|
| Local       | Declared inside a function, accessible only within it. |
| Global      | Declared outside functions, accessible throughout the script. |
| Enclosing   | Found in nested functions (nonlocal). |
| Built-in    | Predefined names in Python (e.g., `print`, `len`). |

- Example of scope:  
  ```python
  x = 10  # Global variable

  def func():
      y = 5  # Local variable
      print(y)

  print(x)  # Accessible
  print(y)  # Error: y is not defined outside the function
  ```  

### Global and Nonlocal Keywords  
- `global`: Used inside a function to modify a global variable.  
  ```python
  x = 10

  def change():
      global x
      x = 20
  ```  
- `nonlocal`: Used inside a nested function to modify a variable in an enclosing scope.  
  ```python
  def outer():
      x = 5
      def inner():
          nonlocal x
          x = 10
  ```  

### Data Types  
- Python automatically assigns types:  
  | Type | Example |
  |------|---------|
  | `int` | `x = 10` |
  | `float` | `y = 3.14` |
  | `str` | `name = "Alice"` |
  | `bool` | `flag = True` |
  | `list` | `lst = [1, 2, 3]` |
  | `tuple` | `tup = (1, 2, 3)` |
  | `set` | `s = {1, 2, 3}` |
  | `dict` | `d = {"key": "value"}` |

### Type Conversion  
| Function | Description | Example |
|----------|------------|---------|
| `int()` | Converts to integer | `int("10") → 10` |
| `float()` | Converts to float | `float("3.14") → 3.14` |
| `str()` | Converts to string | `str(100) → "100"` |
| `list()` | Converts to list | `list((1, 2, 3)) → `[1, 2, 3]` |
| `tuple()` | Converts to tuple | `tuple([1, 2, 3]) → `(1, 2, 3)` |

### Mutable vs Immutable Variables  
| Type | Mutable? | Example |
|------|---------|---------|
| `list` | ✅ Yes | `[1, 2, 3] → lst[0] = 5` |
| `dict` | ✅ Yes | `{"a": 1} → d["a"] = 2` |
| `set` | ✅ Yes | `{1, 2, 3} → s.add(4)` |
| `int` | ❌ No | `x = 10 → x = 20` (creates a new object) |
| `float` | ❌ No | `y = 3.14 → y = 4.5` |
| `str` | ❌ No | `"hello" → "world"` (new string created) |
| `tuple` | ❌ No | `(1, 2, 3) → cannot modify` |

### Deleting Variables  
- Use `del` to remove a variable:  
  ```python
  x = 10
  del x  # Now x is undefined
  ```  

### Special Variables  
- Underscore `_` as a temporary or unused variable:  
  ```python
  _, y = (1, 2)  # Ignoring first value
  ```  
- `__name__`: Holds the module name.  
- `__file__`: Holds the filename in a script.  

### Constants  
- No built-in constant type, but conventionally written in uppercase:  
  ```python
  PI = 3.14159
  MAX_USERS = 1000
  ```  