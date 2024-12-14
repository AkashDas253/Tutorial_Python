
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