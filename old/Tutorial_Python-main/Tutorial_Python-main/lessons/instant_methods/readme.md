## **Instance Methods in Python (OOP)**

---

### **Overview**

* **Instance methods** are the **most common** type of method in Python classes.
* They are **bound to the instance** of the class and can **access and modify instance variables**.
* They must accept `self` as the **first parameter**, which refers to the **current object**.

---

### **Syntax**

```python
class MyClass:
    def instance_method(self, arg1, arg2):
        # Access or modify instance variables
        self.data = arg1 + arg2
        print(self.data)
```

---

### **Example**

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def display_info(self):  # Instance method
        print(f"Name: {self.name}, Marks: {self.marks}")

s1 = Student("Alice", 85)
s1.display_info()  # Name: Alice, Marks: 85
```

---

### **Key Features**

| Feature         | Description                  |
| --------------- | ---------------------------- |
| Bound to        | Object instance              |
| Requires `self` | ✅ Yes                        |
| Can access      | Instance and class variables |
| Can modify      | Instance variables only      |
| Invoked using   | `object_name.method()`       |

---

### **Modifying Instance State**

```python
class BankAccount:
    def __init__(self, balance):
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount  # Modifies instance variable
        print("Deposited:", amount)

    def show_balance(self):
        print("Balance:", self.balance)

acc = BankAccount(1000)
acc.deposit(500)
acc.show_balance()  # Balance: 1500
```

---

### **Difference from Other Method Types**

| Method Type     | First Parameter | Access to `self` | Bound To  |
| --------------- | --------------- | ---------------- | --------- |
| Instance Method | `self`          | ✅                | Object    |
| Class Method    | `cls`           | ❌                | Class     |
| Static Method   | None            | ❌                | Not bound |

---

### **When to Use**

* When the method needs to:

  * Access or modify instance-specific data
  * Behave differently for each object
  * Be overridden in child classes (supports polymorphism)

---
