
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