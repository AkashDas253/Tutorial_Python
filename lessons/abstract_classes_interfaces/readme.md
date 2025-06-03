## **Abstract Classes and Interfaces in Python**

---

### **Overview**

* Both **abstract classes** and **interfaces** are used to define common methods that must be implemented by subclasses.
* Python uses the `abc` (Abstract Base Class) module to create abstract classes.
* Python does not have built-in interfaces like some other languages, but abstract classes with only abstract methods can act as interfaces.

---

### **Abstract Classes**

---

#### **Definition**

* A class that **cannot be instantiated directly**.
* Can contain **abstract methods** (methods without implementation).
* Subclasses **must override all abstract methods**.
* Used to define a **common API** for related classes.

---

#### **Creating Abstract Classes**

* Import `ABC` and `abstractmethod` from `abc`.
* Inherit from `ABC`.
* Use `@abstractmethod` decorator for abstract methods.

---

#### **Syntax**

```python
from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    @abstractmethod
    def my_method(self):
        pass
```

---

#### **Example**

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass

class Dog(Animal):
    def sound(self):
        return "Bark"

class Cat(Animal):
    def sound(self):
        return "Meow"

# a = Animal()  # Error: Can't instantiate abstract class
d = Dog()
print(d.sound())  # Output: Bark
```

---

### **Interfaces (Conceptual)**

---

* Python does not have explicit interface syntax.
* An **interface** is typically an abstract class with **only abstract methods** and no implemented methods.
* Used to specify a contract that classes must follow.

---

### **Differences Between Abstract Class and Interface**

| Feature              | Abstract Class                            | Interface (Conceptual)        |
| -------------------- | ----------------------------------------- | ----------------------------- |
| Instantiation        | Cannot instantiate                        | Cannot instantiate            |
| Methods              | Can have abstract and concrete methods    | Only abstract methods         |
| Multiple Inheritance | Supported                                 | Supported                     |
| Purpose              | Base class with some common functionality | Define a strict contract      |
| Attributes           | Can have instance variables               | Usually no instance variables |

---

### **Summary**

| Concept        | Key Points                                                                              |
| -------------- | --------------------------------------------------------------------------------------- |
| Abstract Class | Class with `@abstractmethod`, cannot instantiate directly, can have implemented methods |
| Interface      | Abstract class with only abstract methods, defines a contract                           |
| Usage          | Ensure subclasses implement certain methods                                             |

---
