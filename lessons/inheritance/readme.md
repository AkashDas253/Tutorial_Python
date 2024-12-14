
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

