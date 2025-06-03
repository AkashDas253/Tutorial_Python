## **Class and Object in Python**

---

### **Class**

A **class** is a user-defined blueprint for creating objects. It defines the structure (attributes) and behavior (methods) that the objects created from the class will have.

#### **Syntax**

```python
class ClassName:
    def __init__(self, args):
        # initialize attributes
        self.attr1 = value
    
    def method_name(self, args):
        # method logic
```

---

### **Object**

An **object** is an instance of a class. It holds actual data and uses methods defined in the class to operate on that data.

#### **Syntax**

```python
obj = ClassName(arguments)
```

---

### **Attributes**

* **Instance Attributes**: Created inside `__init__`, unique to each object.
* **Class Attributes**: Defined directly inside the class but outside methods, shared by all instances.

---

### **Methods**

* Defined inside a class to define behaviors.
* The first argument is always `self` (refers to the object itself).

#### **Types**

| Method Type     | Decorator       | First Parameter |
| --------------- | --------------- | --------------- |
| Instance Method | None            | `self`          |
| Class Method    | `@classmethod`  | `cls`           |
| Static Method   | `@staticmethod` | None            |

---

### **The `self` Keyword**

* Refers to the instance of the class
* Used to access instance variables and methods

---

### **Constructor**

* Special method `__init__()` used to initialize object attributes when created.

---

### **Example**

```python
class Student:
    school = "ABC School"  # class attribute
    
    def __init__(self, name, grade):
        self.name = name      # instance attribute
        self.grade = grade
    
    def display(self):
        print(f"{self.name} is in grade {self.grade}")
```

```python
s1 = Student("Alice", 10)
s2 = Student("Bob", 9)
s1.display()
s2.display()
```

---

### **Memory Allocation**

* **Each object** gets its own copy of instance attributes.
* **Class attributes** are shared and stored once in memory.

---

### **Accessing Members**

| Type            | Access Syntax         |
| --------------- | --------------------- |
| Attribute       | `object.attr_name`    |
| Method          | `object.method()`     |
| Class Attribute | `ClassName.attr_name` |

---
