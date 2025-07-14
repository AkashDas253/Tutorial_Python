## **Tuple (`tuple`) in Python**  

### **Definition**  
- A **tuple** is an **ordered**, **immutable**, and **iterable** collection of elements.  
- Defined using parentheses `()` or `tuple()`.  
- Can store multiple data types, including nested tuples.  

```python
my_tuple = (1, "apple", 3.5, (2, 3))
print(my_tuple)  # (1, 'apple', 3.5, (2, 3))
```

---

### **Tuple Indexing & Slicing**  
| Operation | Example | Result |
|-----------|---------|--------|
| Indexing | `t[0]` | `1` |
| Negative Indexing | `t[-1]` | `(2, 3)` |
| Slicing | `t[1:3]` | `('apple', 3.5)` |
| Step Slicing | `t[::2]` | `(1, 3.5)` |
| Reverse Tuple | `t[::-1]` | `((2, 3), 3.5, 'apple', 1)` |

```python
t = (1, "apple", 3.5, (2, 3))
print(t[1])  # apple
print(t[-1])  # (2, 3)
print(t[1:3])  # ('apple', 3.5)
print(t[::-1])  # ((2, 3), 3.5, 'apple', 1)
```

---

### **Tuple Operations**  
| Operator | Example | Result |
|----------|---------|--------|
| Concatenation (`+`) | `(1, 2) + (3, 4)` | `(1, 2, 3, 4)` |
| Repetition (`*`) | `(1, 2) * 3` | `(1, 2, 1, 2, 1, 2)` |
| Membership (`in`) | `3 in (1, 2, 3)` | `True` |
| Length (`len()`) | `len((1, 2, 3))` | `3` |

```python
print((1, 2) + (3, 4))  # (1, 2, 3, 4)
print((1, 2) * 3)  # (1, 2, 1, 2, 1, 2)
print(3 in (1, 2, 3))  # True
print(len((1, 2, 3)))  # 3
```

---

### **Tuple Methods**  
| Method | Description | Example | Result |
|--------|------------|---------|--------|
| `count(x)` | Counts occurrences of `x` | `t.count(2)` | `1` |
| `index(x)` | Returns index of `x` | `t.index(3)` | `2` |

```python
t = (1, 2, 3, 2, 4)
print(t.count(2))  # 2
print(t.index(3))  # 2
```

---

### **Tuple Packing & Unpacking**  
```python
a, b, c = (1, 2, 3)
print(a, b, c)  # 1 2 3
```

```python
# Swapping values using tuple unpacking
x, y = 10, 20
x, y = y, x
print(x, y)  # 20 10
```

---

### **Nested Tuples**  
```python
nested = (1, (2, 3), (4, (5, 6)))
print(nested[2][1][1])  # 6
```

---

### **Converting Other Data Types to Tuple**
| Function | Example | Result |
|----------|---------|--------|
| `tuple(list)` | `tuple([1, 2, 3])` | `(1, 2, 3)` |
| `tuple(set)` | `tuple({1, 2, 3})` | `(1, 2, 3)` |
| `tuple(string)` | `tuple("abc")` | `('a', 'b', 'c')` |

```python
print(tuple("hello"))  # ('h', 'e', 'l', 'l', 'o')
print(tuple([1, 2, 3]))  # (1, 2, 3)
```

---
---

## Tuples

### In-Depth Note on Python Tuples

#### Properties of Tuples

- **Ordered**: Tuples maintain the order of elements.
- **Immutable**: Once created, the elements of a tuple cannot be changed.
- **Indexed**: Elements can be accessed using indices.
- **Heterogeneous**: Tuples can contain elements of different data types.
- **Hashable**: Tuples can be used as keys in dictionaries if they contain only hashable types.

#### Creating Tuples

1. **Empty Tuple**
   ```python
   my_tuple = ()
   ```

2. **Tuple with One Element** (note the comma)
   ```python
   my_tuple = (element,)
   ```

3. **Tuple with Multiple Elements**
   ```python
   my_tuple = (element1, element2, element3)
   ```

4. **Tuple Without Parentheses** (tuple packing)
   ```python
   my_tuple = element1, element2, element3
   ```

5. **Using `tuple` Constructor**
   ```python
   my_tuple = tuple([element1, element2, element3])
   ```

#### Accessing Elements

1. **Using Indexing**
   ```python
   element = my_tuple[0]
   ```

2. **Using Negative Indexing**
   ```python
   element = my_tuple[-1]
   ```

3. **Using Slicing**
   ```python
   sub_tuple = my_tuple[1:3]
   ```

#### Tuple Operations

1. **Concatenation**
   ```python
   new_tuple = my_tuple1 + my_tuple2
   ```

2. **Repetition**
   ```python
   repeated_tuple = my_tuple * 3
   ```

3. **Membership Test**
   ```python
   if element in my_tuple:
       # Code to execute if element is in my_tuple
   ```

4. **Iterating Over Elements**
   ```python
   for element in my_tuple:
       print(element)
   ```

#### Tuple Methods

1. **`count` Method**: Returns the number of occurrences of a specified value
   ```python
   count = my_tuple.count(element)
   ```

2. **`index` Method**: Returns the index of the first occurrence of a specified value
   ```python
   index = my_tuple.index(element)
   ```

#### Unpacking Tuples

1. **Basic Unpacking**
   ```python
   a, b, c = my_tuple
   ```

2. **Unpacking with `*` Operator**
   ```python
   a, *b, c = my_tuple
   ```

#### Nested Tuples

1. **Creating Nested Tuples**
   ```python
   nested_tuple = (element1, (element2, element3), element4)
   ```

2. **Accessing Elements in Nested Tuples**
   ```python
   inner_element = nested_tuple[1][0]
   ```

#### Tuple Comprehension (Not Directly Supported, Use Generator Expression)

1. **Using Generator Expression**
   ```python
   my_tuple = tuple(x*x for x in range(5))
   ```

### Tuple Methods

- `tuple.count(value)` - Returns the number of times a specified value occurs in a tuple. `value` is the element to count.
- `tuple.index(value, start=0, end=len(tuple))` - Searches the tuple for a specified value and returns the position of where it was found. `value` is the element to search for, `start` and `end` specify the range to search within.