## **Dictionary (`dict`) in Python**  

### **Definition**  
- A **dictionary** is an **unordered**, **mutable**, and **iterable** collection of **key-value pairs**.  
- Keys must be **unique** and **immutable** (strings, numbers, or tuples), while values can be any data type.  
- Defined using curly braces `{}` or the `dict()` constructor.  

```python
my_dict = {"name": "Alice", "age": 25, "city": "New York"}
print(my_dict)  # {'name': 'Alice', 'age': 25, 'city': 'New York'}
```

---

### **Dictionary Access**  
| Operation | Example | Result |
|-----------|---------|--------|
| Access Value | `d["key"]` | Value |
| `get(key, default)` | `d.get("age", 30)` | `25` |
| Check Key | `"name" in d` | `True` |

```python
d = {"name": "Alice", "age": 25, "city": "New York"}
print(d["name"])  # Alice
print(d.get("age", 30))  # 25
print("city" in d)  # True
```

---

### **Dictionary Modification**  
| Operation | Example | Result |
|-----------|---------|--------|
| Add Item | `d["gender"] = "Female"` | `{'name': 'Alice', 'age': 25, 'city': 'New York', 'gender': 'Female'}` |
| Update Value | `d["age"] = 26` | `{'name': 'Alice', 'age': 26, 'city': 'New York'}` |
| Remove Key | `del d["age"]` | Removes `'age'` |
| `pop(key, default)` | `d.pop("city")` | `'New York'` |
| `popitem()` | `d.popitem()` | Removes last inserted key-value pair |
| `clear()` | `d.clear()` | `{}` |

```python
d["gender"] = "Female"
d["age"] = 26
print(d.pop("city"))  # New York
print(d)  # {'name': 'Alice', 'age': 26, 'gender': 'Female'}
```

---

### **Dictionary Methods**  
| Method | Description | Example | Result |
|--------|------------|---------|--------|
| `keys()` | Returns all keys | `d.keys()` | `dict_keys(['name', 'age'])` |
| `values()` | Returns all values | `d.values()` | `dict_values(['Alice', 25])` |
| `items()` | Returns key-value pairs | `d.items()` | `dict_items([('name', 'Alice'), ('age', 25)])` |
| `update(dict)` | Merges dictionaries | `d.update({"city": "LA"})` | `{'name': 'Alice', 'age': 25, 'city': 'LA'}` |

```python
print(d.keys())  # dict_keys(['name', 'age'])
print(d.values())  # dict_values(['Alice', 25])
print(d.items())  # dict_items([('name', 'Alice'), ('age', 25)])
```

---

### **Dictionary Comprehension**  
```python
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
```

---

### **Nested Dictionary**  
```python
students = {
    "Alice": {"age": 25, "city": "New York"},
    "Bob": {"age": 22, "city": "Los Angeles"}
}
print(students["Alice"]["age"])  # 25
```

---

### **Converting Other Data Types to Dictionary**  
| Function | Example | Result |
|----------|---------|--------|
| `dict(list of tuples)` | `dict([(1, "one"), (2, "two")])` | `{1: "one", 2: "two"}` |
| `dict(zip(keys, values))` | `dict(zip(["a", "b"], [1, 2]))` | `{'a': 1, 'b': 2}` |

```python
print(dict(zip(["a", "b"], [1, 2])))  # {'a': 1, 'b': 2}
```

---
---


## Dictionaries

#### Properties of Dictionaries

- **Unordered**: Dictionaries are unordered collections of items.
- **Mutable**: Dictionaries can be changed after creation.
- **Indexed**: Dictionaries are indexed by keys.
- **Keys**: Keys must be unique and immutable (e.g., strings, numbers, tuples).
- **Values**: Values can be of any data type and can be `duplicated` and `nested`.


#### Creating a Dictionary

1. **Empty Dictionary**
   ```python
   my_dict = {}
   ```

2. **Dictionary with Initial Values**
   ```python
   my_dict = {
       "key1": "value1",
       "key2": "value2",
       "key3": "value3"
   }
   ```

#### Accessing Values

1. **Using Keys**
   ```python
   value = my_dict["key1"]
   ```

2. **Using `get` Method**
   ```python
   value = my_dict.get("key1")
   ```

#### Adding and Updating Values

1. **Adding a New Key-Value Pair**
   ```python
   my_dict["new_key"] = "new_value"
   ```

2. **Updating an Existing Key-Value Pair**
   ```python
   my_dict["key1"] = "updated_value"
   ```

#### Removing Values

1. **Using `del` Statement**
   ```python
   del my_dict["key1"]
   ```

2. **Using `pop` Method**
   ```python
   value = my_dict.pop("key2")
   ```

3. **Using `popitem` Method** (removes the last inserted key-value pair)
   ```python
   key, value = my_dict.popitem()
   ```

4. **Using `clear` Method** (removes all items)
   ```python
   my_dict.clear()
   ```

#### Dictionary Operations

1. **Checking if a Key Exists**
   ```python
   if "key1" in my_dict:
       # Code to execute if key1 exists in my_dict
   ```

2. **Iterating Over Keys**
   ```python
   for key in my_dict:
       print(key)
   ```

3. **Iterating Over Values**
   ```python
   for value in my_dict.values():
       print(value)
   ```

4. **Iterating Over Key-Value Pairs**
   ```python
   for key, value in my_dict.items():
       print(key, value)
   ```

5. **Dictionary Comprehension**
   ```python
   squared_numbers = {x: x*x for x in iterable}
   ```

#### Dictionary Methods

1. **`keys` Method**: Returns a view object of all keys
   ```python
   keys = my_dict.keys()
   ```

2. **`values` Method**: Returns a view object of all values
   ```python
   values = my_dict.values()
   ```

3. **`items` Method**: Returns a view object of all key-value pairs
   ```python
   items = my_dict.items()
   ```

4. **`update` Method**: Updates the dictionary with elements from another dictionary or an iterable of key-value pairs
   ```python
   my_dict.update({"key4": "value4", "key5": "value5"})
   ```

5. **`copy` Method**: Returns a shallow copy of the dictionary
   ```python
   new_dict = my_dict.copy()
   ```

6. **`fromkeys` Method**: Creates a new dictionary with keys from an iterable and values set to a specified value
   ```python
   keys = ["a", "b", "c"]
   new_dict = dict.fromkeys(keys, 0)
   ```

7. **`setdefault` Method**: Returns the value of a key if it is in the dictionary; if not, inserts the key with a specified value
   ```python
   value = my_dict.setdefault("key6", "default_value")
   ```


### Dictionary Methods

- `dict.clear()` - Removes all the elements from the dictionary.
- `dict.copy()` - Returns a copy of the dictionary.
- `dict.fromkeys(keys, value=None)` - Returns a dictionary with the specified keys and value. `keys` is an iterable of keys, and `value` is the value to set for all keys (default is None).
- `dict.get(key, default=None)` - Returns the value of the specified key. `key` is the key to look up, and `default` is the value to return if the key is not found (default is None).
- `dict.items()` - Returns a view object that displays a list of a dictionary's key-value tuple pairs.
- `dict.keys()` - Returns a view object that displays a list of all the keys in the dictionary.
- `dict.pop(key, default=None)` - Removes the element with the specified key. `key` is the key to remove, and `default` is the value to return if the key is not found (default is None).
- `dict.popitem()` - Removes the last inserted key-value pair.
- `dict.setdefault(key, default=None)` - Returns the value of the specified key. If the key does not exist, inserts the key with the specified value. `key` is the key to look up, and `default` is the value to set if the key is not found (default is None).
- `dict.update([other])` - Updates the dictionary with the specified key-value pairs. `other` can be another dictionary or an iterable of key-value pairs.
- `dict.values()` - Returns a view object that displays a list of all the values in the dictionary.
