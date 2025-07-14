## **Instance Variables in Python (OOP)**

---

### **Overview**

* **Instance variables** are variables **defined inside a class** and **bound to the instance** of the class.
* They hold **data unique to each object**.
* Created using the `self` keyword inside methods (typically in `__init__()`).

---

### **Declaration**

```python
class Student:
    def __init__(self, name, marks):
        self.name = name        # Instance variable
        self.marks = marks      # Instance variable
```

Each object will have **its own copy** of `name` and `marks`.

---

### **Accessing Instance Variables**

```python
s1 = Student("Alice", 90)
s2 = Student("Bob", 85)

print(s1.name)  # Alice
print(s2.name)  # Bob
```

---

### **Modifying Instance Variables**

```python
s1.marks = 95
print(s1.marks)  # 95
print(s2.marks)  # 85 (unchanged)
```

* You can also add **new instance variables** at runtime:

```python
s1.grade = 'A'
print(s1.grade)  # A
# print(s2.grade) → AttributeError
```

---

### **Deleting Instance Variables**

```python
del s1.name
# print(s1.name) → AttributeError
```

---

### **Where Instance Variables Can Be Created**

| Location                      | Creation Allowed |
| ----------------------------- | ---------------- |
| Inside `__init__()`           | ✅ Common         |
| Inside other instance methods | ✅ Allowed        |
| Outside class (using object)  | ✅ Allowed        |

---

### **Example**

```python
class Car:
    def __init__(self, brand):
        self.brand = brand  # Instance variable

    def set_model(self, model):
        self.model = model  # Another instance variable

c = Car("Toyota")
c.set_model("Corolla")
print(c.brand, c.model)  # Toyota Corolla
```

---

### **Comparison with Class Variables**

| Feature             | Instance Variable | Class Variable                        |
| ------------------- | ----------------- | ------------------------------------- |
| Bound to            | Object            | Class                                 |
| Declared using      | `self.var_name`   | Class body (not in method)            |
| Shared between objs | ❌ No              | ✅ Yes                                 |
| Accessed via        | `object.var_name` | `Class.var_name` or `object.var_name` |

---
