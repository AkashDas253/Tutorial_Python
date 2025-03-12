## **Iterators in Python**  

An **iterator** is an object that allows traversal through elements of a collection (e.g., lists, tuples) **one at a time**. It follows the **Iterator Protocol**, which requires:  
- `__iter__()` → Returns the iterator object itself  
- `__next__()` → Returns the next element or raises `StopIteration` when finished  

---

## **1. Iterables vs. Iterators**  

| Feature | Iterable | Iterator |
|---------|---------|----------|
| Definition | An object that can return an iterator | An object that produces values one at a time |
| Methods | `__iter__()` | `__iter__()`, `__next__()` |
| Examples | Lists, Tuples, Strings, Sets, Dictionaries | Objects created using `iter()` |

### **Example (Iterable vs. Iterator)**
```python
nums = [1, 2, 3]  # Iterable
it = iter(nums)    # Iterator

print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
# print(next(it))  # Raises StopIteration
```

---

## **2. Creating a Custom Iterator**  

A custom iterator must define both `__iter__()` and `__next__()`.

### **Example (Custom Iterator)**
```python
class Counter:
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

count = Counter(1, 5)
for num in count:
    print(num)
```

**Output:**  
```
1
2
3
4
5
```

---

## **3. `iter()` with Sentinel Value**  
`iter()` can be used with a callable function and a sentinel (stop value).

### **Example (`iter()` with Sentinel)**
```python
import random

def random_number():
    return random.randint(1, 10)

for num in iter(random_number, 7):  # Stops when 7 is generated
    print(num)
```

---

## **4. Using Iterators with Built-in Functions**  

| Function | Description |
|----------|------------|
| `map()` | Applies a function to each item |
| `filter()` | Filters items based on condition |
| `zip()` | Combines multiple iterables |
| `enumerate()` | Returns index-value pairs |

### **Example (`zip()` and `enumerate()`)**
```python
names = ["Alice", "Bob"]
ages = [25, 30]

for name, age in zip(names, ages):
    print(name, age)

for index, value in enumerate(names):
    print(index, value)
```

---

## **5. `iter()` on File Objects**  
File objects are iterators and can be read line by line.

### **Example (Reading File Line by Line)**
```python
with open("file.txt", "r") as file:
    for line in iter(file.readline, ''):
        print(line.strip())
```

---

---
---

## Python Iterators

### Python Iterators

- An iterator is an object that contains a countable number of values.
- It can be iterated upon, meaning you can traverse through all the values.
- An iterator implements the iterator protocol, which consists of the methods `__iter__()` and `__next__()`.

### Iterator vs Iterable

- Lists, tuples, dictionaries, and sets are iterable objects.
- These objects have an `iter()` method to get an iterator.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
```

- Strings are also iterable objects, containing a sequence of characters.

#### Syntax
```python
mystr = "banana"
myit = iter(mystr)

print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
```

### Looping Through an Iterator

- Use a `for` loop to iterate through an iterable object.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)
```

#### Syntax
```python
mystr = "banana"

for x in mystr:
  print(x)
```

- The `for` loop creates an iterator object and executes the `next()` method for each loop.

### Create an Iterator

- Implement the methods `__iter__()` and `__next__()` to create an iterator.

#### Syntax
```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    x = self.a
    self.a += 1
    return x

myclass = MyNumbers()
myiter = iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
```

### StopIteration

- Use the `StopIteration` statement to prevent the iteration from going on forever.

#### Syntax
```python
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    if self.a <= 20:
      x = self.a
      self.a += 1
      return x
    else:
      raise StopIteration

myclass = MyNumbers()
myiter = iter(myclass)

for x in myiter:
  print(x)
```