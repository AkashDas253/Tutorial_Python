# Multilevel Inheritance in Python

### Concept

* Multilevel inheritance is when a class is derived from a **child class**, which itself is derived from a **parent class**.
* Forms a **chain of inheritance**: Grandparent → Parent → Child.
* Methods and attributes are inherited down the chain.

---

### Syntax

```python
class Grandparent:
    def greet(self):
        print("Hello from Grandparent")

class Parent(Grandparent):
    def welcome(self):
        print("Welcome from Parent")

class Child(Parent):
    def info(self):
        print("Info from Child")

# Create object
c = Child()
c.greet()    # Inherited from Grandparent
c.welcome()  # Inherited from Parent
c.info()     # Defined in Child
```

---

### Constructor in Multilevel Inheritance

```python
class Grandparent:
    def __init__(self):
        print("Grandparent constructor")

class Parent(Grandparent):
    def __init__(self):
        super().__init__()  # Call Grandparent constructor
        print("Parent constructor")

class Child(Parent):
    def __init__(self):
        super().__init__()  # Call Parent constructor
        print("Child constructor")

Child()
# Output:
# Grandparent constructor
# Parent constructor
# Child constructor
```

---

### Method Overriding in Multilevel

```python
class Parent:
    def greet(self):
        print("Parent greet")

class Child(Parent):
    def greet(self):  # Override Parent method
        print("Child greet")

c = Child()
c.greet()  # Child greet
```

* Child can **override parent methods**.
* Use `super()` to call the parent’s version if needed.

---

### Key Points

* Forms a **linear chain of inheritance**.
* Supports **method and constructor overriding**.
* Use `super()` to propagate method calls up the chain.
* Simpler than multiple inheritance, **no ambiguity**.

---
