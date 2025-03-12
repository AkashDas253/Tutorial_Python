## Data Types in Python  

### Classification of Data Types  
| Type | Mutable? | Example |
|------|---------|---------|
| **Numeric** | ❌ No | `int`, `float`, `complex` |
| **Boolean** | ❌ No | `bool` (`True`, `False`) |
| **Sequence** | ✅/❌ Mixed | `str` (❌), `list` (✅), `tuple` (❌), `range` (❌) |
| **Set** | ✅ Yes | `set`, `frozenset` (❌) |
| **Mapping** | ✅ Yes | `dict` |
| **Binary** | ✅ Yes | `bytes` (❌), `bytearray` (✅), `memoryview` (✅) |

---

### **1. Numeric Types**  
| Type | Description | Example |
|------|------------|---------|
| `int` | Integer (no decimal) | `x = 10` |
| `float` | Floating-point number | `y = 3.14` |
| `complex` | Complex number | `z = 2 + 3j` |

- Type Conversion:  
  ```python
  int(3.5)  # 3
  float(10)  # 10.0
  complex(5)  # (5+0j)
  ```

---

### **2. Boolean Type**  
- Represents `True` or `False`.  
- Internally, `True = 1`, `False = 0`.  
  ```python
  bool(1)  # True
  bool(0)  # False
  bool([])  # False (empty objects are False)
  ```

---

### **3. Sequence Types**  
| Type | Mutable? | Example |
|------|---------|---------|
| `str` | ❌ No | `"Hello"` |
| `list` | ✅ Yes | `[1, 2, 3]` |
| `tuple` | ❌ No | `(1, 2, 3)` |
| `range` | ❌ No | `range(5) → 0,1,2,3,4` |

#### **String (`str`)**
- Immutable sequence of characters.
  ```python
  text = "Hello"
  text[0]  # 'H'
  ```

#### **List (`list`)**
- Mutable sequence of elements.
  ```python
  lst = [1, 2, 3]
  lst[0] = 10  # Allowed
  ```

#### **Tuple (`tuple`)**
- Immutable sequence.
  ```python
  tup = (1, 2, 3)
  tup[0] = 10  # Error
  ```

#### **Range (`range`)**
- Immutable sequence of numbers.
  ```python
  range(5)  # 0, 1, 2, 3, 4
  ```

---

### **4. Set Types**  
| Type | Mutable? | Example |
|------|---------|---------|
| `set` | ✅ Yes | `{1, 2, 3}` |
| `frozenset` | ❌ No | `frozenset({1, 2, 3})` |

- No duplicate elements.
- Unordered collection.
  ```python
  s = {1, 2, 3}
  s.add(4)  # Allowed
  fs = frozenset(s)  # Immutable
  ```

---

### **5. Mapping Type**  
| Type | Mutable? | Example |
|------|---------|---------|
| `dict` | ✅ Yes | `{"a": 1, "b": 2}` |

- Key-value pairs.
  ```python
  d = {"name": "Alice", "age": 25}
  d["age"] = 26  # Allowed
  ```

---

### **6. Binary Types**  
| Type | Mutable? | Example |
|------|---------|---------|
| `bytes` | ❌ No | `b"hello"` |
| `bytearray` | ✅ Yes | `bytearray(5)` |
| `memoryview` | ✅ Yes | `memoryview(bytes(5))` |

- Used for handling binary data like images or files.
  ```python
  b = bytes([65, 66, 67])  # b'ABC'
  ba = bytearray(b)
  ba[0] = 68  # Mutates to b'DBC'
  ```

---

### **Type Checking and Conversion**
| Function | Description | Example |
|----------|------------|---------|
| `type(x)` | Returns type of `x` | `type(10) → int` |
| `isinstance(x, type)` | Checks if `x` is of given type | `isinstance(10, int) → True` |
| `str(x)` | Converts to string | `str(100) → "100"` |
| `int(x)` | Converts to integer | `int("10") → 10` |
| `float(x)` | Converts to float | `float("3.14") → 3.14` |
| `list(x)` | Converts to list | `list((1, 2, 3)) → [1, 2, 3]` |
| `tuple(x)` | Converts to tuple | `tuple([1, 2, 3]) → (1, 2, 3)` |
| `set(x)` | Converts to set | `set([1, 2, 2]) → {1, 2}` |

---

### **Mutable vs Immutable Types**
| Type | Mutable? |
|------|---------|
| `list` | ✅ Yes |
| `dict` | ✅ Yes |
| `set` | ✅ Yes |
| `bytearray` | ✅ Yes |
| `int` | ❌ No |
| `float` | ❌ No |
| `str` | ❌ No |
| `tuple` | ❌ No |
| `frozenset` | ❌ No |
| `bytes` | ❌ No |

---
---


# Datatypes:

## Built-in Data Types:

In programming, data type is an important concept.

Variables can store data of different types, and different types can do different things.

Python has the following data types built-in by default, in these categories:

- **Text Type**: `str`
- **Numeric Types**: `int`, `float`, `complex`
- **Sequence Types**: `list`, `tuple`, `range`
- **Mapping Type**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean Type**: `bool`
- **Binary Types**: `bytes`, `bytearray`, `memoryview`
- **None Type**: `NoneType`

### Getting the Data Type: type()

You can get the data type of any object by using the `type()` function:

#### Example
```python
x = 5
print(type(x))  # Output: <class 'int'>
```

### Setting the Data Type

In Python, the data type is set when you assign a value to a variable:

#### Example
```python
x = "Hello World"  # str
x = 20  # int
x = 20.5  # float
x = 1j  # complex
x = ["apple", "banana", "cherry"]  # list
x = ("apple", "banana", "cherry")  # tuple
x = range(6)  # range
x = {"name": "John", "age": 36}  # dict
x = {"apple", "banana", "cherry"}  # set
x = frozenset({"apple", "banana", "cherry"})  # frozenset
x = True  # bool
x = b"Hello"  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
x = None  # NoneType
```

### Setting the Specific Data Type

If you want to specify the data type, you can use the following constructor functions:

#### Example
```python
x = str("Hello World")  # str
x = int(20)  # int
x = float(20.5)  # float
x = complex(1j)  # complex
x = list(("apple", "banana", "cherry"))  # list
x = tuple(("apple", "banana", "cherry"))  # tuple
x = range(6)  # range
x = dict(name="John", age=36)  # dict
x = set(("apple", "banana", "cherry"))  # set
x = frozenset(("apple", "banana", "cherry"))  # frozenset
x = bool(5)  # bool
x = bytes(5)  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
```

### Type Casting

Type casting is the process of converting one data type to another. Python supports several built-in functions for type casting.

#### Example
```python
# Convert to integer
x = int(1)  # x will be 1
y = int(2.8)  # y will be 2
z = int("3")  # z will be 3

# Convert to float
a = float(1)  # a will be 1.0
b = float(2.8)  # b will be 2.8
c = float("3")  # c will be 3.0
d = float("4.2")  # d will be 4.2

# Convert to string
e = str(1)  # e will be '1'
f = str(2.8)  # f will be '2.8'
g = str(3)  # g will be '3'

# Convert to list
h = list((1, 2, 3))  # h will be [1, 2, 3]

# Convert to tuple
i = tuple([1, 2, 3])  # i will be (1, 2, 3)

# Convert to set
j = set([1, 2, 3])  # j will be {1, 2, 3}

# Convert to dictionary
k = dict([(1, 'one'), (2, 'two')])  # k will be {1: 'one', 2: 'two'}

# Convert to boolean
l = bool(1)  # l will be True
m = bool(0)  # m will be False
```

### Summary of Type Casting Functions

- `int(x)`: Converts `x` to an integer.
- `float(x)`: Converts `x` to a float.
- `str(x)`: Converts `x` to a string.
- `list(x)`: Converts `x` to a list.
- `tuple(x)`: Converts `x` to a tuple.
- `set(x)`: Converts `x` to a set.
- `dict(x)`: Converts `x` to a dictionary.
- `bool(x)`: Converts `x` to a boolean.

---
