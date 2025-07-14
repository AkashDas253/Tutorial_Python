## **Method Resolution Order (MRO) and Conflict in Python**

---

### **Overview**

* MRO defines the order in which base classes are searched when executing a method.
* Crucial for **multiple inheritance** to determine which method to call.
* Helps resolve **method conflicts** when multiple parent classes have methods with the same name.
* Python uses the **C3 linearization algorithm** to compute MRO.

---

### **Method Resolution Order (MRO)**

---

#### **Key Points**

* Python searches methods in the order defined by MRO.
* MRO can be accessed using the class attribute: `ClassName.__mro__` or the built-in function `mro()`.
* Ensures a consistent, monotonic order respecting inheritance hierarchy.

---

#### **Example**

```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):
    pass

d = D()
d.show()  # Output: B
print(D.__mro__)
```

* Output of `d.show()` is `B.show` because `B` appears before `C` in MRO.
* `D.__mro__` shows the order: `(D, B, C, A, object)`

---

### **MRO Calculation: C3 Linearization**

* Combines parent classes in order while preserving the local precedence order.
* Ensures subclasses come before superclasses.
* Prevents inconsistencies in method lookup.

---

### **Conflict in Multiple Inheritance**

---

#### **What is a Conflict?**

* Occurs when multiple base classes have methods or attributes with the same name.
* Without MRO, Python would not know which method to invoke.

---

#### **How Python Resolves Conflict**

* Uses MRO to determine the order of method lookup.
* Calls the first method found according to MRO.

---

#### **Diamond Problem**

* Common conflict case when a class inherits from two classes that both inherit from a common base.
* Python’s MRO resolves diamond problem correctly.

```python
class A:
    def show(self):
        print("A")

class B(A):
    def show(self):
        print("B")

class C(A):
    def show(self):
        print("C")

class D(B, C):
    pass

d = D()
d.show()  # Calls B.show(), resolves diamond conflict using MRO
```

---

### **Summary**

| Concept         | Description                                   |
| --------------- | --------------------------------------------- |
| MRO             | Order Python follows to resolve methods       |
| Access MRO      | `ClassName.__mro__` or `ClassName.mro()`      |
| Conflict        | Multiple methods with same name in parents    |
| Resolution      | Use MRO to pick method from first found class |
| Diamond Problem | Resolved by MRO ensuring consistent lookup    |

---
