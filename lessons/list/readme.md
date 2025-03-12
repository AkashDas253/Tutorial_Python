## **List (`list`) in Python**  

### **Definition**  
- A **list** is an **ordered**, **mutable**, and **iterable** collection of elements.  
- Can store multiple data types, including other lists (nested lists).  
- Defined using square brackets `[]`.  

```python
my_list = [1, "apple", 3.5, [2, 3]]
print(my_list)  # [1, 'apple', 3.5, [2, 3]]
```

---

### **List Indexing & Slicing**  
| Operation | Example | Result |
|-----------|---------|--------|
| Indexing | `my_list[0]` | `1` |
| Negative Indexing | `my_list[-1]` | `[2, 3]` |
| Slicing | `my_list[1:3]` | `['apple', 3.5]` |
| Step Slicing | `my_list[::2]` | `[1, 3.5]` |
| Reverse List | `my_list[::-1]` | `[[2, 3], 3.5, 'apple', 1]` |

```python
my_list = [1, "apple", 3.5, [2, 3]]
print(my_list[1])  # apple
print(my_list[-1])  # [2, 3]
print(my_list[1:3])  # ['apple', 3.5]
print(my_list[::-1])  # [[2, 3], 3.5, 'apple', 1]
```

---

### **List Operations**  
| Operator | Example | Result |
|----------|---------|--------|
| Concatenation (`+`) | `[1, 2] + [3, 4]` | `[1, 2, 3, 4]` |
| Repetition (`*`) | `[1, 2] * 3` | `[1, 2, 1, 2, 1, 2]` |
| Membership (`in`) | `3 in [1, 2, 3]` | `True` |
| Length (`len()`) | `len([1, 2, 3])` | `3` |

```python
print([1, 2] + [3, 4])  # [1, 2, 3, 4]
print([1, 2] * 3)  # [1, 2, 1, 2, 1, 2]
print(3 in [1, 2, 3])  # True
print(len([1, 2, 3]))  # 3
```

---

### **List Methods**  
| Method | Description | Example | Result |
|--------|------------|---------|--------|
| `append(x)` | Adds `x` to the end | `l.append(4)` | `[1, 2, 3, 4]` |
| `extend(lst)` | Adds multiple elements | `l.extend([4, 5])` | `[1, 2, 3, 4, 5]` |
| `insert(i, x)` | Inserts `x` at index `i` | `l.insert(1, "a")` | `[1, "a", 2, 3]` |
| `remove(x)` | Removes first occurrence of `x` | `l.remove(2)` | `[1, 3]` |
| `pop(i)` | Removes and returns item at index `i` | `l.pop(1)` | `2` |
| `index(x)` | Returns index of `x` | `l.index(3)` | `2` |
| `count(x)` | Counts occurrences of `x` | `l.count(2)` | `1` |
| `sort()` | Sorts list (ascending) | `l.sort()` | `[1, 2, 3]` |
| `reverse()` | Reverses the list | `l.reverse()` | `[3, 2, 1]` |
| `copy()` | Returns a copy | `l.copy()` | `[1, 2, 3]` |
| `clear()` | Removes all elements | `l.clear()` | `[]` |

```python
l = [1, 2, 3]
l.append(4)  # [1, 2, 3, 4]
l.extend([5, 6])  # [1, 2, 3, 4, 5, 6]
l.insert(2, "x")  # [1, 2, "x", 3, 4, 5, 6]
l.remove(2)  # [1, "x", 3, 4, 5, 6]
print(l.pop(2))  # 3, list is now [1, "x", 4, 5, 6]
print(l.index("x"))  # 1
print(l.count(4))  # 1
l.sort()  # TypeError (due to mixed types)
l = [3, 1, 2]
l.sort()  # [1, 2, 3]
l.reverse()  # [3, 2, 1]
l_copy = l.copy()  # [3, 2, 1]
l.clear()  # []
```

---

### **List Comprehension**  
- A compact way to create lists.

```python
nums = [x for x in range(5)]  # [0, 1, 2, 3, 4]
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

---

### **List Unpacking**  
```python
a, b, c = [1, 2, 3]
print(a, b, c)  # 1 2 3
```

---

### **Nested Lists**  
```python
matrix = [[1, 2], [3, 4]]
print(matrix[0][1])  # 2
```

---

### **Converting Other Data Types to List**
| Function | Example | Result |
|----------|---------|--------|
| `list(tuple)` | `list((1, 2, 3))` | `[1, 2, 3]` |
| `list(set)` | `list({1, 2, 3})` | `[1, 2, 3]` |
| `list(string)` | `list("abc")` | `['a', 'b', 'c']` |

```python
print(list("hello"))  # ['h', 'e', 'l', 'l', 'o']
print(list((1, 2, 3)))  # [1, 2, 3]
```

---
---

## List:

### Properties and usage of Python Lists

#### Properties of Lists

- **Ordered**: Lists maintain the order of elements.
- **Mutable**: Lists can be changed after creation.
- **Indexed**: Lists are indexed by integers, starting from 0.
- **Heterogeneous**: Lists can contain elements of different data types.

#### Creating a List

1. **Empty List**
   ```python
   my_list = []
   ```

2. **List with Initial Values**
   ```python
   my_list = [1, 2, 3, "four", 5.0]
   ```

#### Accessing Elements

1. **Using Index**
   ```python
   element = my_list[0]  # Access the first element
   ```

2. **Using Negative Index**
   ```python
   element = my_list[-1]  # Access the last element
   ```

#### Adding Elements

1. **Using `append` Method** (adds an element to the end)
   ```python
   my_list.append("new_element")
   ```

2. **Using `insert` Method** (inserts an element at a specified position)
   ```python
   my_list.insert(1, "inserted_element")
   ```

3. **Using `extend` Method** (extends the list by appending elements from an iterable)
   ```python
   my_list.extend([6, 7, 8])
   ```

#### Removing Elements

1. **Using `remove` Method** (removes the first occurrence of a value)
   ```python
   my_list.remove("four")
   ```

2. **Using `pop` Method** (removes and returns the element at a specified position)
   ```python
   element = my_list.pop(2)
   ```

3. **Using `clear` Method** (removes all elements)
   ```python
   my_list.clear()
   ```

#### List Operations

1. **Checking if an Element Exists**
   ```python
   if "new_element" in my_list:
       # Code to execute if "new_element" exists in my_list
   ```

2. **Iterating Over Elements**
   ```python
   for element in my_list:
       print(element)
   ```

3. **List Comprehension**
   ```python
   squared_numbers = [x*x for x in range(5)]
   ```

4. **Slicing a List**
   ```python
   sub_list = my_list[1:3]  # Get elements from index 1 to 2
   ```

#### List Methods

1. **`index` Method**: Returns the index of the first occurrence of a value
   ```python
   index = my_list.index("new_element")
   ```

2. **`count` Method**: Returns the number of occurrences of a value
   ```python
   count = my_list.count(2)
   ```

3. **`sort` Method**: Sorts the list in ascending order
   ```python
   my_list.sort()
   ```

4. **`reverse` Method**: Reverses the elements of the list
   ```python
   my_list.reverse()
   ```

5. **`copy` Method**: Returns a shallow copy of the list
   ```python
   new_list = my_list.copy()
   ```

#### List Functions

1. **`len` Function**: Returns the number of elements in the list
   ```python
   length = len(my_list)
   ```

2. **`max` Function**: Returns the largest element in the list
   ```python
   maximum = max(my_list)
   ```

3. **`min` Function**: Returns the smallest element in the list
   ```python
   minimum = min(my_list)
   ```

4. **`sum` Function**: Returns the sum of all elements in the list
   ```python
   total = sum(my_list)
   ```

### List Methods

- `list.append(element)` - Adds an element at the end of the list. `element` is the item to be added.
- `list.clear()` - Removes all the elements from the list.
- `list.copy()` - Returns a copy of the list.
- `list.count(value)` - Returns the number of elements with the specified value. `value` is the item to be counted.
- `list.extend(iterable)` - Adds the elements of a list (or any iterable) to the end of the current list. `iterable` is the collection of elements to be added.
- `list.index(value, start=0, end=len(list))` - Returns the index of the first element with the specified value. `value` is the item to search for, `start` and `end` specify the range to search within.
- `list.insert(index, element)` - Adds an element at the specified position. `index` is the position to insert the element, and `element` is the item to be added.
- `list.pop(index=-1)` - Removes the element at the specified position. `index` is the position of the element to be removed (default is the last item).
- `list.remove(value)` - Removes the item with the specified value. `value` is the item to be removed.
- `list.reverse()` - Reverses the order of the list.
- `list.sort(key=None, reverse=False)` - Sorts the list. `key` is a function that serves as a key for the sort comparison, and `reverse` is a boolean value to sort in descending order.
