### Object-Oriented Programming (OOP) Syntax in Python

#### Class and Object

```python
# Define a class
class ClassName:
    def __init__(self, attribute1, attribute2):
        self.attribute1 = attribute1
        self.attribute2 = attribute2

    def method_name(self):
        # Method implementation
        pass

# Create an object
object_name = ClassName(attribute1_value, attribute2_value)
```

#### Inheritance

```python
# Base class
class BaseClass:
    def __init__(self, attribute):
        self.attribute = attribute

    def base_method(self):
        # Method implementation
        pass

# Derived class
class DerivedClass(BaseClass):
    def __init__(self, attribute, additional_attribute):
        super().__init__(attribute)
        self.additional_attribute = additional_attribute

    def derived_method(self):
        # Method implementation
        pass
```

#### Encapsulation

```python
# Define a class with private attributes
class EncapsulatedClass:
    def __init__(self, public_attribute, private_attribute):
        self.public_attribute = public_attribute
        self.__private_attribute = private_attribute  # Private attribute

    def get_private_attribute(self):
        return self.__private_attribute  # Accessor method

    def __private_method(self):
        # Private method implementation
        pass
```

#### Polymorphism

```python
# Define classes with a common interface
class ClassA:
    def common_method(self):
        return "ClassA implementation"

class ClassB:
    def common_method(self):
        return "ClassB implementation"

# Polymorphic function
def polymorphic_function(obj):
    print(obj.common_method())

# Using polymorphism
obj_a = ClassA()
obj_b = ClassB()
polymorphic_function(obj_a)  # Output: ClassA implementation
polymorphic_function(obj_b)  # Output: ClassB implementation
```

### Summary Table

| Concept       | Syntax Example                                                                 |
|---------------|--------------------------------------------------------------------------------|
| Class         | `class ClassName: ...`                                                         |
| Object        | `object_name = ClassName(attribute1_value, attribute2_value)`                  |
| Inheritance   | `class DerivedClass(BaseClass): ...`                                           |
| Encapsulation | `self.__private_attribute = private_attribute`                                 |
| Polymorphism  | `def polymorphic_function(obj): print(obj.common_method())`                    |

This syntax overview provides a quick reference for implementing OOP concepts in Python.


### Summary Table

| Concept       | Description                                                                 | Example                                                                 |
|---------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Class         | Blueprint for creating objects                                              | `class Dog: ...`                                                        |
| Object        | Instance of a class                                                         | `my_dog = Dog("Buddy", 3)`                                              |
| Inheritance   | New class inherits attributes and methods from an existing class            | `class Dog(Animal): ...`                                                |
| Encapsulation | Bundling data and methods within one unit, using private variables/methods  | `self.__make = make`                                                    |
| Polymorphism  | Presenting the same interface for different underlying forms (data types)   | `def make_it_fly(entity): print(entity.fly())`                          |

This concise overview covers the fundamental concepts of OOP in Python, providing examples and a summary table for quick reference.