## **Object-Oriented Programming (OOP) in Python**  

### **Definition**  
- **Object-Oriented Programming (OOP)** is a programming paradigm based on the concept of **objects**, which contain **data** (attributes) and **methods** (functions).  
- Python supports OOP through **classes and objects**.  

---

### **Key OOP Concepts**  
| Concept | Description | Example |
|---------|------------|---------|
| **Class** | Blueprint for creating objects | `class Car:` |
| **Object** | Instance of a class | `my_car = Car()` |
| **Encapsulation** | Hiding data using private/protected attributes | `self.__speed = 60` |
| **Abstraction** | Hiding complex implementation details | Abstract methods using `ABC` |
| **Inheritance** | A class inherits properties from another | `class ElectricCar(Car):` |
| **Polymorphism** | Same method behaves differently in different classes | Method Overriding |

---

### **1. Class and Object**  
- **Class**: Defines a blueprint for objects.  
- **Object**: An instance of a class with attributes and methods.  

```python
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def display(self):
        print(f"Car: {self.brand} {self.model}")

# Creating an object
my_car = Car("Tesla", "Model S")
my_car.display()  # Car: Tesla Model S
```

---

### **2. Encapsulation**  
- Restricts access to certain details of an object.  
- Uses **private (`__`)** and **protected (`_`)** attributes.  

```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Private variable

    def get_balance(self):
        return self.__balance  # Accessing private variable

account = BankAccount(1000)
print(account.get_balance())  # 1000
# print(account.__balance)  # Error: AttributeError
```

---

### **3. Abstraction**  
- Hides implementation details using abstract classes and methods.  
- Uses `ABC` module.  

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14 * self.radius * self.radius

c = Circle(5)
print(c.area())  # 78.5
```

---

### **4. Inheritance**  
- One class derives properties from another.  

| Type | Description | Example |
|------|------------|---------|
| **Single** | One class inherits another | `class B(A):` |
| **Multiple** | Inherits from multiple classes | `class C(A, B):` |
| **Multilevel** | Inherits from a derived class | `class C(B):` (B → C) |
| **Hierarchical** | One class is a parent to multiple | `class B(A), class C(A)` |
| **Hybrid** | Combination of types | Multiple + Multilevel |

```python
class Vehicle:
    def __init__(self, brand):
        self.brand = brand

    def show_brand(self):
        print(f"Brand: {self.brand}")

class Car(Vehicle):  # Single Inheritance
    def __init__(self, brand, model):
        super().__init__(brand)
        self.model = model

my_car = Car("Tesla", "Model 3")
my_car.show_brand()  # Brand: Tesla
```

---

### **5. Polymorphism**  
- **Method Overriding**: Redefining a method in a derived class.  

```python
class Animal:
    def sound(self):
        print("Animal makes a sound")

class Dog(Animal):
    def sound(self):
        print("Bark")

d = Dog()
d.sound()  # Bark
```

- **Method Overloading** (Python does not support it directly, but can be achieved using default arguments).  

```python
class Math:
    def add(self, a, b, c=0):
        return a + b + c

m = Math()
print(m.add(2, 3))  # 5
print(m.add(2, 3, 4))  # 9
```

---

### **6. Special Methods (`__init__`, `__str__`, etc.)**  
- Python provides **dunder (double underscore) methods** to define behavior.  

| Method | Purpose |
|--------|---------|
| `__init__` | Constructor (initializes object) |
| `__str__` | String representation of object |
| `__repr__` | Official string representation |
| `__len__` | Defines length behavior |
| `__getitem__` | Allows indexing |

```python
class Person:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"Person: {self.name}"

p = Person("Alice")
print(p)  # Person: Alice
```

---

### **7. Multiple Inheritance**  
- A class inherits from multiple base classes.  

```python
class A:
    def show(self):
        print("Class A")

class B:
    def display(self):
        print("Class B")

class C(A, B):
    pass

obj = C()
obj.show()  # Class A
obj.display()  # Class B
```

---

### **8. Class vs Instance Variables**  
| Type | Defined | Shared? | Accessed via |
|------|--------|---------|-------------|
| **Instance Variable** | `self.variable` in `__init__` | No | `self.variable` |
| **Class Variable** | Outside `__init__`, inside class | Yes | `ClassName.variable` |

```python
class Employee:
    company = "TechCorp"  # Class Variable

    def __init__(self, name):
        self.name = name  # Instance Variable

e1 = Employee("Alice")
e2 = Employee("Bob")

print(e1.company, e1.name)  # TechCorp Alice
print(e2.company, e2.name)  # TechCorp Bob
```

---

### **9. Static and Class Methods**  
| Method | Decorator | Purpose |
|--------|----------|---------|
| **Instance Method** | None | Works on instance variables |
| **Class Method** | `@classmethod` | Works on class variables |
| **Static Method** | `@staticmethod` | No access to class or instance variables |

```python
class Example:
    var = "class variable"

    @classmethod
    def class_method(cls):
        print(f"Class method: {cls.var}")

    @staticmethod
    def static_method():
        print("Static method")

Example.class_method()  # Class method: class variable
Example.static_method()  # Static method
```

---

### **10. Object Comparison (`__eq__`, `__lt__`, etc.)**  
| Method | Operator | Purpose |
|--------|---------|---------|
| `__eq__` | `==` | Equality |
| `__ne__` | `!=` | Not Equal |
| `__lt__` | `<` | Less Than |
| `__gt__` | `>` | Greater Than |
| `__le__` | `<=` | Less or Equal |
| `__ge__` | `>=` | Greater or Equal |

```python
class Number:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

n1 = Number(5)
n2 = Number(5)
print(n1 == n2)  # True
```

---

---
---

## Object-Oriented Programming (OOP) Syntax in Python

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