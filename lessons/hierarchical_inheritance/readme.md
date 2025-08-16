# Hierarchical Inheritance in Python

### Concept

* Hierarchical inheritance occurs when **multiple child classes inherit from a single parent class**.
* Each child class can have its own methods and can also access parent class methods.
* Useful when several classes share common functionality from a single parent.

---

### Syntax

```python
class Parent:
    def greet(self):
        print("Hello from Parent")

class Child1(Parent):
    def info1(self):
        print("Info from Child1")

class Child2(Parent):
    def info2(self):
        print("Info from Child2")

# Create objects
c1 = Child1()
c2 = Child2()

c1.greet()   # Inherited from Parent
c1.info1()   # Child1 specific
c2.greet()   # Inherited from Parent
c2.info2()   # Child2 specific
```

---

### Constructor in Hierarchical Inheritance

```python
class Parent:
    def __init__(self):
        print("Parent constructor")

class Child1(Parent):
    def __init__(self):
        super().__init__()  # Call Parent constructor
        print("Child1 constructor")

class Child2(Parent):
    def __init__(self):
        super().__init__()  # Call Parent constructor
        print("Child2 constructor")

Child1()
# Output:
# Parent constructor
# Child1 constructor

Child2()
# Output:
# Parent constructor
# Child2 constructor
```

---

### Method Overriding

```python
class Parent:
    def greet(self):
        print("Parent greet")

class Child1(Parent):
    def greet(self):  # Override Parent method
        print("Child1 greet")

Child1().greet()  # Child1 greet
```

* Each child can **override parent methods independently**.
* Use `super()` to call parent method if required.

---

### Key Points

* Single parent, **multiple children**.
* Supports **method and constructor overriding** in each child.
* Common parent methods are **reused** in all child classes.
* Avoids duplication of common functionality.

---
