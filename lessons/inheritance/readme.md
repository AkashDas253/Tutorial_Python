## **Inheritance in Python**

---

### **Definition**

**Inheritance** is an OOP feature that allows one class (**child/subclass**) to acquire the properties and behaviors (attributes and methods) of another class (**parent/superclass**). It promotes **code reuse**, **extensibility**, and **polymorphism**.

---

### **Purpose**

* Reuse existing code
* Create hierarchical relationships
* Enable polymorphism
* Promote DRY (Don't Repeat Yourself) principle

---

### **Basic Syntax**

```python
class Parent:
    def method(self):
        print("Parent method")

class Child(Parent):
    pass
```

---

### **Types of Inheritance**

| Type         | Description                               | Syntax Pattern                    |
| ------------ | ----------------------------------------- | --------------------------------- |
| Single       | One child inherits from one parent        | `class B(A):`                     |
| Multiple     | One child inherits from multiple parents  | `class C(A, B):`                  |
| Multilevel   | Inheritance chain of multiple levels      | `class C(B):` where `class B(A):` |
| Hierarchical | Multiple children inherit from one parent | `class B(A):`, `class C(A):`      |
| Hybrid       | Combines multiple inheritance types       | Mix of above                      |

---

### **Types of Inheritance with Syntax**

#### **Single Inheritance**

```python
class A:
    def show(self):
        print("A")

class B(A):
    def display(self):
        print("B")
```

#### **Multilevel Inheritance**

```python
class A:
    def show(self):
        print("A")

class B(A):
    def display(self):
        print("B")

class C(B):
    def greet(self):
        print("C")
```

#### **Multiple Inheritance**

```python
class A:
    def show(self):
        print("A")

class B:
    def display(self):
        print("B")

class C(A, B):
```

---

### **Using `super()`**

* Used to call a method from the parent class inside the child class.
* Commonly used inside constructors (`__init__`).

```python
class Parent:
    def __init__(self):
        print("Parent constructor")

class Child(Parent):
    def __init__(self):
        super().__init__()
        print("Child constructor")
```

---

### **Method Overriding**

Child class can override methods of the parent class with the same name.

```python
class Parent:
    def greet(self):
        print("Hello from Parent")

class Child(Parent):
    def greet(self):
        print("Hello from Child")
```

---

### **Constructor Overriding**

If a child class defines its own `__init__`, the parent’s `__init__` is not automatically called unless `super().__init__()` is used.

---

### **Attribute Resolution Order (MRO)**

Python uses **Method Resolution Order (MRO)** to determine which method to execute first in case of multiple inheritance.

Use:

```python
ClassName.__mro__
```

---

### **Example**

```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

d = Dog()
d.speak()  # Output: Dog barks
```

---

### **Benefits**

* Simplifies code by reusing existing logic
* Encourages hierarchical class design
* Supports polymorphism and runtime method resolution

---
