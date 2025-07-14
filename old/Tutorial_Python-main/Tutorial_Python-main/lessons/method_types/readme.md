## **Method Types in Python**

---

### **Overview**

In Python, methods inside a class can be of three types based on their behavior and how they are called:

* **Instance methods**
* **Class methods**
* **Static methods**

---

### **1. Instance Methods**

---

#### **Definition**

* Regular methods that take `self` as the first parameter.
* Operate on an instance of the class.
* Can access and modify **instance attributes** and **other instance methods**.

---

#### **Syntax**

```python
class MyClass:
    def instance_method(self, args):
        # Access instance data using self
        pass
```

---

#### **Example**

```python
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, my name is {self.name}")

p = Person("Alice")
p.greet()  # Output: Hello, my name is Alice
```

---

### **2. Class Methods**

---

#### **Definition**

* Methods that take `cls` as the first parameter instead of `self`.
* Operate on the **class itself**, not on instances.
* Can access and modify **class-level attributes**.
* Defined with the `@classmethod` decorator.

---

#### **Syntax**

```python
class MyClass:
    @classmethod
    def class_method(cls, args):
        # Access class data using cls
        pass
```

---

#### **Example**

```python
class Person:
    population = 0

    def __init__(self, name):
        self.name = name
        Person.population += 1

    @classmethod
    def get_population(cls):
        return cls.population

print(Person.get_population())  # Output: 0
p1 = Person("Alice")
p2 = Person("Bob")
print(Person.get_population())  # Output: 2
```

---

### **3. Static Methods**

---

#### **Definition**

* Methods that do not take `self` or `cls` parameters.
* Behave like regular functions but belong to the class namespace.
* Do not modify object or class state.
* Defined with the `@staticmethod` decorator.

---

#### **Syntax**

```python
class MyClass:
    @staticmethod
    def static_method(args):
        # Does not access instance or class data
        pass
```

---

#### **Example**

```python
class Math:
    @staticmethod
    def add(x, y):
        return x + y

print(Math.add(5, 3))  # Output: 8
```

---

### **Comparison Tables**

| Method Type     | First Parameter | Accesses      | Decorator       | Typical Use Case                   |
| --------------- | --------------- | ------------- | --------------- | ---------------------------------- |
| Instance Method | `self`          | Instance data | None            | Methods acting on object state     |
| Class Method    | `cls`           | Class data    | `@classmethod`  | Factory methods, class-wide logic  |
| Static Method   | None            | Neither       | `@staticmethod` | Utility functions related to class |

| Method Type   | First Parameter | Access to Instance Variables | Access to Class Variables | Decorator       | Typical Use Cases                |
| ------------- | --------------- | ---------------------------- | ------------------------- | --------------- | -------------------------------- |
| Class Method  | `cls`           | No                           | Yes                       | `@classmethod`  | Factory methods, class-wide data |
| Static Method | None            | No                           | No                        | `@staticmethod` | Utility/helper functions         |


---

### **Summary**

* Use **instance methods** to work with individual objects.
* Use **class methods** when you need to work with the class itself.
* Use **static methods** for utility or helper functions that don’t access instance or class data.

---
