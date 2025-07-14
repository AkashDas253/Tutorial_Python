## **Constructor and Destructor in Python**

---

### **Constructor**

---

#### **Definition**

A **constructor** is a special method automatically called when an object is created. It initializes the object’s attributes.

---

#### **Constructor Method Name**

* In Python, the constructor method is `__init__`.

---

#### **Syntax**

```python
class MyClass:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
```

---

#### **Purpose**

* Initialize object state
* Set default or passed-in values
* Allocate resources if needed

---

#### **Example**

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Alice", 30)
print(p.name)  # Output: Alice
print(p.age)   # Output: 30
```

---

### **Destructor**

---

#### **Definition**

A **destructor** is a special method invoked when an object is about to be destroyed (garbage collected). It is used to perform cleanup activities like releasing resources.

---

#### **Destructor Method Name**

* In Python, the destructor method is `__del__`.

---

#### **Syntax**

```python
class MyClass:
    def __del__(self):
        print("Destructor called, object deleted")
```

---

#### **Usage Notes**

* Python uses automatic garbage collection; explicit destructor use is rare.
* `__del__` is called when the reference count of the object reaches zero.
* The timing of `__del__` call is unpredictable in some implementations like PyPy.
* Avoid complex logic in destructors to prevent resource leaks or errors.

---

#### **Example**

```python
class FileHandler:
    def __init__(self, filename):
        self.file = open(filename, 'r')

    def __del__(self):
        self.file.close()
        print("File closed")

fh = FileHandler("data.txt")
del fh  # Destructor called here
```

---

### **Key Differences Between Constructor and Destructor**

| Feature     | Constructor (`__init__`)          | Destructor (`__del__`)               |
| ----------- | --------------------------------- | ------------------------------------ |
| Purpose     | Initialize object attributes      | Clean up before object destruction   |
| Called When | Object creation                   | Object deletion (garbage collection) |
| Mandatory   | Not mandatory but common practice | Optional and rarely used             |
| Arguments   | Can accept parameters             | No parameters                        |

---
