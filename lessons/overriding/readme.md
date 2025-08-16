# Overriding in Python 

**Overriding** means providing a **new implementation in a subclass** or redefining the behavior of existing methods, constructors, operators, or properties.
It is a key part of **inheritance** and **polymorphism** in Python.

---

## Types of Overriding

### Method Overriding

Subclass defines a method with the **same name and parameters** as in parent class.
When called on the child object, the child’s version is executed.

```python
class Parent:
    def show(self):
        print("Parent show()")

class Child(Parent):
    def show(self):   # Override
        print("Child show()")

Child().show()  # Child show()
```

---

### Constructor Overriding (`__init__`)

Child class defines its own constructor, overriding parent’s.
Parent constructor can still be called with `super()`.

```python
class Parent:
    def __init__(self):
        print("Parent init")

class Child(Parent):
    def __init__(self):
        super().__init__()
        print("Child init")

Child()
```

---

### Dunder (Magic Method) Overriding

Python allows overriding **special methods** to customize object behavior.

| Method                  | Purpose               |
| ----------------------- | --------------------- |
| `__str__`               | String (`str(obj)`)   |
| `__repr__`              | Debug representation  |
| `__len__`               | Length (`len(obj)`)   |
| `__getitem__`           | Indexing (`obj[key]`) |
| `__setitem__`           | Assignment            |
| `__iter__` / `__next__` | Iteration             |
| `__call__`              | Callable objects      |
| `__eq__`, `__lt__`...   | Comparisons           |
| `__add__`, `__sub__`... | Arithmetic            |
| `__enter__`, `__exit__` | Context managers      |

```python
class MyList:
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data) * 2  # Override len()

print(len(MyList([1,2,3])))  # 6
```

---

### Property Overriding

Subclass can redefine **class variables**, **instance variables**, or **properties**.

```python
class Parent:
    value = 10

class Child(Parent):
    value = 20  # Overrides property

print(Child().value)  # 20
```

With `@property`:

```python
class Parent:
    @property
    def data(self): return "Parent data"

class Child(Parent):
    @property
    def data(self): return "Child data"

print(Child().data)  # Child data
```

---

### Operator Overriding

Overriding **dunder methods** changes how operators work.

```python
class Vector:
    def __init__(self,x,y): self.x,self.y=x,y
    def __add__(self,other): return Vector(self.x+other.x, self.y+other.y)
    def __str__(self): return f"({self.x},{self.y})"

print(Vector(1,2) + Vector(3,4))  # (4,6)
```

---

### Abstract Method Overriding

Required when subclassing an **abstract base class** (`abc`).

```python
from abc import ABC, abstractmethod
class Shape(ABC):
    @abstractmethod
    def area(self): pass

class Circle(Shape):
    def area(self): return 3.14*5*5
```

---

### MRO-based Overriding (Multiple Inheritance)

In multiple inheritance, subclasses can override methods of any parent.
Python decides using **Method Resolution Order (C3 linearization)**.

```python
class A: 
    def show(self): print("A")
class B(A): 
    def show(self): print("B")
class C(A): 
    def show(self): print("C")
class D(B, C): pass

D().show()  # B (MRO: D → B → C → A)
```

---

## Overriding vs Overloading (Comparison)

| Aspect         | Overloading                                   | Overriding                  |
| -------------- | --------------------------------------------- | --------------------------- |
| Where          | Same class                                    | Subclass modifies parent    |
| Signature      | Same name, different parameters               | Same name, same parameters  |
| Purpose        | Multiple forms of function/operator           | Redefine inherited behavior |
| Python Support | Limited (`*args`, defaults, `singledispatch`) | Fully supported             |

---

## Summary

* **Method Overriding** → redefine methods in subclass.
* **Constructor Overriding** → redefine `__init__`.
* **Dunder Method Overriding** → customize built-in operations (`__str__`, `__len__`, etc.).
* **Property Overriding** → subclass redefines parent’s properties or variables.
* **Operator Overriding** → redefine operator behavior (`+`, `==`, etc.).
* **Abstract Method Overriding** → required when subclass inherits abstract base class.
* **MRO-based Overriding** → in multiple inheritance, Python uses MRO to decide resolution.

---
