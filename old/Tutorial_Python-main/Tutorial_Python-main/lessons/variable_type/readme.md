## **Variable Types in Python (OOP Context)**

---

### **Overview**

In Object-Oriented Programming (OOP) with Python, variables are categorized based on their **scope** and **binding**:

* **Instance Variables**
* **Class Variables**
* **Local Variables**
* **Global Variables**

Each type has different visibility, lifetime, and use-case.

---

### **1. Instance Variables**

* Belong to **object instances**.
* Defined using `self` inside methods.
* Unique to each object.

```python
class Student:
    def __init__(self, name):
        self.name = name  # Instance variable

s1 = Student("Alice")
s2 = Student("Bob")
print(s1.name)  # Alice
print(s2.name)  # Bob
```

---

### **2. Class Variables**

* Shared by **all instances** of a class.
* Defined **inside the class** but **outside any method**.
* Accessed using `ClassName.var` or `self.__class__.var`.

```python
class Student:
    school = "XYZ School"  # Class variable

    def __init__(self, name):
        self.name = name

s1 = Student("Alice")
s2 = Student("Bob")
print(s1.school)  # XYZ School
print(s2.school)  # XYZ School
Student.school = "ABC School"
print(s1.school)  # ABC School
```

---

### **3. Local Variables**

* Defined **inside methods or functions**.
* Accessible **only within** the method scope.

```python
class Student:
    def show(self):
        grade = "A"  # Local variable
        print(grade)

s = Student()
s.show()
# print(grade) → Error: grade is not defined
```

---

### **4. Global Variables**

* Declared **outside all classes and functions**.
* Accessible everywhere using the `global` keyword (if modified inside a function).

```python
school = "XYZ"

class Student:
    def show(self):
        print("School:", school)

s = Student()
s.show()
```

---

### **5. Constant Variable (Convention)**

* Python doesn't have true constants.
* Use **uppercase names** to indicate that a variable should not be changed.

```python
PI = 3.14159
```

---

### **Comparison Table**

| Type              | Scope           | Defined In                   | Shared | Accessed Using                          |
| ----------------- | --------------- | ---------------------------- | ------ | --------------------------------------- |
| Instance          | Object-specific | Inside method (via `self`)   | ❌      | `self.var`                              |
| Class             | Class-wide      | Inside class (not in method) | ✅      | `ClassName.var` or `self.__class__.var` |
| Local             | Method/function | Inside method/function       | ❌      | Local scope only                        |
| Global            | Entire module   | Outside all classes          | ✅      | Directly or via `global`                |
| Constant (pseudo) | Global/module   | Uppercase variable           | ✅      | Convention only                         |

---
