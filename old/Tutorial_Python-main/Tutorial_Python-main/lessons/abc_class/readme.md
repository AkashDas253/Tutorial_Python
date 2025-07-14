## **Abstract Base Class (ABC) in Python**

---

### **Overview**

* An **Abstract Base Class (ABC)** is a class that **cannot be instantiated directly**.
* Designed to be **a blueprint for other classes**.
* Defines one or more **abstract methods** that must be **implemented by subclasses**.
* Enforces a common interface for subclasses.

---

### **Module**

* ABCs are implemented using the `abc` module.

```python
from abc import ABC, abstractmethod
```

---

### **Key Concepts**

| Concept                                        | Description                                          |
| ---------------------------------------------- | ---------------------------------------------------- |
| `ABC`                                          | Base class to create abstract classes                |
| `@abstractmethod`                              | Decorator to declare methods that must be overridden |
| Cannot instantiate abstract class directly     |                                                      |
| Subclasses must implement all abstract methods |                                                      |

---

### **Basic Example**

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def sound(self):
        pass  # Abstract method

class Dog(Animal):
    def sound(self):
        return "Bark"

class Cat(Animal):
    def sound(self):
        return "Meow"

# a = Animal()  # Error: Can't instantiate abstract class
dog = Dog()
print(dog.sound())  # Bark
```

---

### **Features**

* Abstract methods enforce **implementation** in subclasses.
* You can have **concrete methods** in ABC that subclasses inherit.
* ABC can have **class variables** and **properties** like normal classes.

---

### **Use Cases**

* Define a **common interface** for related classes.
* Ensure subclasses follow a **contract**.
* Useful in **large projects** and frameworks for consistency.

---

### **Additional Notes**

* You can create **abstract properties** using `@property` and `@abstractmethod` decorators.

```python
class Shape(ABC):
    @property
    @abstractmethod
    def area(self):
        pass
```

* Classes not implementing all abstract methods **cannot be instantiated**.

---
