## **self and super in Python**

---

### **self**

---

#### **Definition**

* `self` is a **reference to the current instance** of a class.
* It allows access to **instance variables** and **methods** within the class.

---

#### **Usage**

* Always the first parameter of instance methods.
* Passed automatically when a method is called on an object.
* Not a keyword, just a strong naming convention (can be named differently but not recommended).

---

#### **Purpose**

* Distinguish between instance attributes and local variables.
* Access or modify object state within methods.
* Call other methods on the same object.

---

#### **Example**

```python
class Person:
    def __init__(self, name):
        self.name = name  # self.name refers to the instance attribute

    def greet(self):
        print(f"Hello, my name is {self.name}")
```

---

### **super**

---

#### **Definition**

* `super()` is a built-in function used to **call a method from a parent (super) class**.
* Useful in inheritance to **access overridden methods** or the parent constructor.

---

#### **Usage**

* Commonly used in subclass methods to extend or reuse functionality from the parent class.
* Automatically resolves the **Method Resolution Order (MRO)** in multiple inheritance.

---

#### **Syntax**

```python
class Parent:
    def greet(self):
        print("Hello from Parent")

class Child(Parent):
    def greet(self):
        super().greet()  # Call parent greet()
        print("Hello from Child")
```

---

#### **Calling Parent Constructor**

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age
```

---

### **Key Points**

| Aspect         | `self`                             | `super()`                                     |
| -------------- | ---------------------------------- | --------------------------------------------- |
| What it refers | Current object instance            | Parent/superclass                             |
| Purpose        | Access instance attributes/methods | Call parent class methods                     |
| Used in        | Instance methods                   | Subclass methods (especially overridden ones) |
| Is a keyword?  | No (convention)                    | Yes (built-in function)                       |

---

### **Common Mistakes**

* Forgetting to include `self` as the first parameter of instance methods.
* Not calling `super()` in subclass constructors, leading to incomplete initialization.
* Using `super()` outside of class inheritance context.

---
