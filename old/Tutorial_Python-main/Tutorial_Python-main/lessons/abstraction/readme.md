## **Abstraction in Python**

---

### **Definition**

**Abstraction** is the OOP concept of **hiding complex internal implementation details** and **exposing only the essential features** of an object to the outside world. It simplifies interface usage and promotes clean code.

---

### **Purpose**

* Hide complexity
* Expose only necessary operations
* Improve modularity
* Enable separation of interface from implementation

---

### **Abstraction vs Encapsulation**

| Feature     | Encapsulation                           | Abstraction                            |
| ----------- | --------------------------------------- | -------------------------------------- |
| Focus       | Data hiding and access control          | Hiding implementation details          |
| Achieved By | Access modifiers and private attributes | Abstract classes and interfaces        |
| Goal        | Protect internal state                  | Simplify usage by hiding complex logic |

---

### **Abstraction in Python**

Python provides abstraction using:

* **Abstract Base Classes (ABCs)** from the `abc` module
* **Interfaces** via method declarations without implementation

---

### **Abstract Base Class (ABC)**

An abstract class **cannot be instantiated** and may contain:

* **Abstract methods** (declared but not implemented)
* **Concrete methods** (fully implemented)

---

### **Syntax**

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    def description(self):
        return "This is a shape"
```

---

### **Creating a Concrete Class**

```python
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius * self.radius
```

---

### **Rules**

* You **must** inherit from `ABC` to define an abstract class.
* You **must** implement all abstract methods in a subclass before instantiation.
* Attempting to instantiate a class with unimplemented abstract methods raises a `TypeError`.

---

### **Use Cases**

* Defining common interfaces for subclasses
* Designing plugins, APIs, or frameworks
* Preventing instantiation of incomplete base classes

---

### **Real-world Analogy**

A **TV remote** offers buttons (interface) but hides internal circuit logic (implementation), allowing users to operate it without understanding how it works.

---
