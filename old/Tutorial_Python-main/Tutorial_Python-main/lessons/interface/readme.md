## **Interface in Python**

---

### **Overview**

* Python **does not have a built-in interface keyword or construct** like some other languages (Java, C#).
* Interfaces are typically implemented using **Abstract Base Classes (ABC)**.
* An **interface defines method signatures** that implementing classes must provide.
* Interfaces **enforce a contract** without implementation details.
* Useful for **polymorphism and design clarity**.

---

### **Implementing Interface Using ABC**

* Define an abstract class with only abstract methods.
* Classes implementing the interface must override all abstract methods.

```python
from abc import ABC, abstractmethod

class VehicleInterface(ABC):

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

class Car(VehicleInterface):

    def start(self):
        print("Car started")

    def stop(self):
        print("Car stopped")

car = Car()
car.start()  # Car started
car.stop()   # Car stopped
```

---

### **Key Points**

| Aspect                     | Description                              |
| -------------------------- | ---------------------------------------- |
| No explicit keyword        | Use ABC with only abstract methods       |
| Enforces method signatures | All methods must be implemented          |
| No method implementation   | Interface methods have no body           |
| Enables polymorphism       | Objects can be treated by interface type |

---

### **Differences Between Abstract Class and Interface in Python**

| Feature                   | Abstract Class                         | Interface                              |
| ------------------------- | -------------------------------------- | -------------------------------------- |
| Can have concrete methods | Yes                                    | No (only abstract methods)             |
| Purpose                   | Base class with partial implementation | Pure contract (only method signatures) |
| Instantiation             | Cannot instantiate                     | Cannot instantiate                     |
| Multiple inheritance      | Supported                              | Supported                              |

---

### **Use Cases**

* Define **API specifications**.
* Enforce **method signatures** across unrelated classes.
* Support **multiple implementations** with a common contract.

---
