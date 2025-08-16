# Multiple Inheritance in Python

### Concept

* Multiple inheritance allows a class to inherit attributes and methods from more than one parent class.
* Useful when combining behaviors from different classes.
* Python uses **Method Resolution Order (MRO)** to determine which parent class’s method is called when conflicts arise.
* MRO follows the **C3 linearization algorithm**.

---

### Syntax

```python
# Multiple Inheritance Example
class Parent1:
    def show(self):
        print("Parent1 show method")

class Parent2:
    def display(self):
        print("Parent2 display method")

class Child(Parent1, Parent2):  # Inherits from both Parent1 and Parent2
    def info(self):
        print("Child info method")

# Create object
obj = Child()
obj.show()     # Inherited from Parent1
obj.display()  # Inherited from Parent2
obj.info()     # Defined in Child
```

---

### Method Resolution Order (MRO)

```python
class A:
    def greet(self):
        print("Hello from A")

class B:
    def greet(self):
        print("Hello from B")

class C(A, B):
    pass

obj = C()
obj.greet()  # "Hello from A" because A comes first in inheritance order

print(C.mro())  
# Output: [<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>]
```

---

### Key Points

* **MRO determines execution order** when multiple parents define the same method.
* Use `super()` to ensure cooperative calls across multiple classes.
* Be careful with diamond problem (when a class inherits from two classes that share a common base).

---
