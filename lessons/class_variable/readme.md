## **Class Variables in Python (OOP)**

---

### **Overview**

* **Class variables** are variables that are **shared across all instances** of a class.
* They are defined **inside the class** but **outside all methods**.
* Belong to the **class itself**, not to any one object.

---

### **Declaration**

```python
class Employee:
    company = "TechCorp"  # Class variable

    def __init__(self, name):
        self.name = name  # Instance variable
```

---

### **Accessing Class Variables**

* Accessed via:

  * Class name: `Employee.company`
  * Instance: `emp1.company` (falls back to class variable if not found in instance)

---

### **Modifying Class Variables**

#### Through Class

```python
Employee.company = "NewCorp"
```

> Affects all instances unless an instance has overridden the variable locally.

#### Through Instance

```python
emp1.company = "InstanceCorp"
```

> Creates a **new instance variable**, does **not modify the class variable**.

---

### **Example**

```python
class Employee:
    company = "TechCorp"  # Class variable

    def __init__(self, name):
        self.name = name

emp1 = Employee("Alice")
emp2 = Employee("Bob")

print(emp1.company)  # TechCorp
print(emp2.company)  # TechCorp

Employee.company = "NewCorp"

print(emp1.company)  # NewCorp
print(emp2.company)  # NewCorp

emp1.company = "LocalCorp"  # New instance variable
print(emp1.company)  # LocalCorp
print(emp2.company)  # NewCorp
```

---

### **Use Cases**

* Keeping count (e.g., number of objects created)
* Shared configuration or constants
* Default values for all instances

---

### **Example: Object Counter**

```python
class Counter:
    count = 0  # Class variable

    def __init__(self):
        Counter.count += 1

a = Counter()
b = Counter()
print(Counter.count)  # 2
```

---

### **Key Points**

| Feature                 | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| Defined in              | Class body (outside methods)                              |
| Shared                  | Yes, among all instances                                  |
| Accessed using          | ClassName or object                                       |
| Modified through class  | Yes, updates all                                          |
| Modified through object | Creates instance variable (doesn’t affect class variable) |

---
