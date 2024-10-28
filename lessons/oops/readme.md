
# OOPs

## Python Classes and Objects

### Python Classes/Objects

- Python is an object-oriented programming language.
- Almost everything in Python is an object, with its properties and methods.
- A Class is like an object constructor, or a "blueprint" for creating objects.

### Create a Class

- To create a class, use the keyword `class`:

#### Syntax
```python
class ClassName:
    property_name = value
```

### Create Object

- Use the class to create objects:

#### Syntax
```python
object_name = ClassName()
print(object_name.property_name)
```

### The `__init__()` Function

- The `__init__()` function is executed when the class is being initiated.
- Use it to assign values to object properties or perform other operations when the object is created.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

object_name = ClassName(value1, value2)
print(object_name.property1)
print(object_name.property2)
```

### The `__str__()` Function

- The `__str__()` function controls the string representation of the class object.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def __str__(self):
        return f"{self.property1}({self.property2})"

object_name = ClassName(value1, value2)
print(object_name)
```

### Object Methods

- Methods in objects are functions that belong to the object.

#### Syntax
```python
class ClassName:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def method_name(self):
        print("Message " + self.property1)

object_name = ClassName(value1, value2)
object_name.method_name()
```

### The `self` Parameter

- The `self` parameter is a reference to the current instance of the class.
- It is used to access variables that belong to the class.
- It can be named anything, but it must be the first parameter of any function in the class.

#### Syntax
```python
class ClassName:
    def __init__(custom_self, param1, param2):
        custom_self.property1 = param1
        custom_self.property2 = param2

    def method_name(custom_self):
        print("Message " + custom_self.property1)

object_name = ClassName(value1, value2)
object_name.method_name()
```

## Python Inheritance

### Inheritance Overview

- Inheritance allows defining a class that inherits all methods and properties from another class.
- **Parent class**: The class being inherited from (base class).
- **Child class**: The class that inherits from another class (derived class).

### Create a Parent Class

- Any class can be a parent class. The syntax is the same as creating any other class.

#### Syntax
```python
class ParentClass:
    def __init__(self, param1, param2):
        self.property1 = param1
        self.property2 = param2

    def method_name(self):
        print(self.property1, self.property2)

# Create an object and execute a method
obj = ParentClass(value1, value2)
obj.method_name()
```

### Create a Child Class

- To create a class that inherits from another class, pass the parent class as a parameter when creating the child class.

#### Syntax
```python
class ChildClass(ParentClass):
    pass

# Create an object and execute an inherited method
obj = ChildClass(value1, value2)
obj.method_name()
```

### Add the `__init__()` Function

- Adding the `__init__()` function to the child class overrides the parent's `__init__()` function.
- To keep the inheritance of the parent's `__init__()` function, call the parent's `__init__()` function within the child's `__init__()`.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2):
        ParentClass.__init__(self, param1, param2)
```

### Use the `super()` Function

- The `super()` function allows the child class to inherit all methods and properties from its parent without explicitly naming the parent class.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2):
        super().__init__(param1, param2)
```

### Add Properties

- Add properties to the child class by defining them in the `__init__()` function.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2, param3):
        super().__init__(param1, param2)
        self.property3 = param3

# Create an object with the new property
obj = ChildClass(value1, value2, value3)
```

### Add Methods

- Add methods to the child class by defining them within the class.
- If a method in the child class has the same name as a method in the parent class, it overrides the parent method.

#### Syntax
```python
class ChildClass(ParentClass):
    def __init__(self, param1, param2, param3):
        super().__init__(param1, param2)
        self.property3 = param3

    def new_method(self):
        print("Welcome", self.property1, self.property2, "to the class of", self.property3)

# Create an object and execute the new method
obj = ChildClass(value1, value2, value3)
obj.new_method()
```

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

## Python Polymorphism

### Definition

- Polymorphism means "many forms".
- In programming, it refers to methods/functions/operators with the same name that can be executed on many objects or classes.

### Function Polymorphism

- Example of a Python function that can be used on different objects: `len()` function.

#### String

- For strings, `len()` returns the number of characters.

#### Syntax
```python
x = "Hello World```"
print(len(x))
```

#### Tuple

- For tuples, `len()` returns the number of items in the tuple.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")
print(len(mytuple))
```

#### Dictionary

- For dictionaries, `len()` returns the number of key/value pairs in the dictionary.

#### Syntax
```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(len(thisdict))
```

### Class Polymorphism

- Polymorphism is often used in class methods, where multiple classes have the same method name.

#### Example

- Different classes with the same method:

#### Syntax
```python
class Car:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Drive```")

class Boat:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Sail```")

class Plane:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  x.move()
```

- The `for` loop at the end demonstrates polymorphism by executing the same method for all three classes.

### Inheritance Class Polymorphism

- Child classes with the same name can use polymorphism.

#### Example

- Create a parent class `Vehicle` and make `Car`, `Boat`, `Plane` child classes of `Vehicle`.

#### Syntax
```python
class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move```")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail```")

class Plane(Vehicle):
  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()
```

- Child classes inherit properties and methods from the parent class.
- The `Car` class is empty but inherits `brand`, `model`, and `move()` from `Vehicle`.
- The `Boat` and `Plane` classes override the `move()` method.
- Polymorphism allows executing the same method for all classes.

## Python Scope

### Scope Definition

- A variable is only available from inside the region it is created. This is called scope.

### Local Scope

- A variable created inside a function belongs to the local scope of that function and can only be used inside that function.

#### Syntax
```python
def myfunc():
    x = 300
    print(x)

myfunc()
```

### Function Inside Function

- A local variable can be accessed from a function within the function.

#### Syntax
```python
def myfunc():
    x = 300
    def myinnerfunc():
        print(x)
    myinnerfunc()

myfunc()
```

### Global Scope

- A variable created in the main body of the Python code is a global variable and belongs to the global scope.
- Global variables are available from within any scope, global and local.

#### Syntax
```python
x = 300

def myfunc():
    print(x)

myfunc()
print(x)
```

### Naming Variables

- If you operate with the same variable name inside and outside of a function, Python will treat them as two separate variables, one available in the global scope (outside the function) and one available in the local scope (inside the function).

#### Syntax
```python
x = 300

def myfunc():
    x = 200
    print(x)

myfunc()
print(x)
```

### Global Keyword

- If you need to create a global variable but are stuck in the local scope, you can use the `global` keyword.
- The `global` keyword makes the variable global.

#### Syntax
```python
def myfunc():
    global x
    x = 300

myfunc()
print(x)
```

- Use the `global` keyword if you want to make a change to a global variable inside a function.

#### Syntax
```python
x = 300

def myfunc():
    global x
    x = 200

myfunc()
print(x)
```

### Nonlocal Keyword

- The `nonlocal` keyword is used to work with variables inside nested functions.
- The `nonlocal` keyword makes the variable belong to the outer function.

#### Syntax
```python
def myfunc1():
    x = "Jane"
    def myfunc2():
        nonlocal x
        x = "hello"
    myfunc2()
    return x

print(myfunc1())
```

