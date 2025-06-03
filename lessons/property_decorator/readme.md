## **Property Decorator in Python**

---

### **Overview**

* The `@property` decorator allows you to **define methods that behave like attributes**.
* It helps achieve **controlled access** to instance variables with **getter, setter, and deleter** functionality.
* Enables **encapsulation** and **data validation** while keeping the syntax simple.

---

### **Using `@property`**

---

#### **Basic Getter**

* Define a method with `@property` to **access a computed or protected attribute like a normal attribute**.

```python
class Person:
    def __init__(self, name):
        self._name = name  # Protected attribute

    @property
    def name(self):
        return self._name

p = Person("Alice")
print(p.name)  # Access like an attribute, calls the getter method
```

---

#### **Setter**

* Use `@<property_name>.setter` to define a setter method that **validates or modifies the attribute** on assignment.

```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value:
            raise ValueError("Name cannot be empty")
        self._name = value

p = Person("Alice")
p.name = "Bob"   # Calls setter
# p.name = ""    # Raises ValueError
```

---

#### **Deleter**

* Use `@<property_name>.deleter` to define a method for deleting the attribute.

```python
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @name.deleter
    def name(self):
        del self._name

p = Person("Alice")
del p.name  # Calls deleter method
```

---

### **Benefits**

* Cleaner syntax to **access or modify attributes** without explicit method calls.
* Allows **encapsulation and validation** while keeping usage simple.
* Avoids breaking existing code when changing attribute to computed property.

---

### **Summary**

| Aspect         | Description                                              |
| -------------- | -------------------------------------------------------- |
| Decorator name | `@property`, `@<property>.setter`, `@<property>.deleter` |
| Purpose        | Encapsulate access to instance variables                 |
| Usage          | Define getter, setter, deleter methods with decorator    |
| Access syntax  | Use like normal attributes (`obj.attr`)                  |

---
