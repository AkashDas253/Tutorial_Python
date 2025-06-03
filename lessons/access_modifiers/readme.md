## **Access Modifiers in Python**

---

### **Definition**

Access modifiers control the **visibility and accessibility** of class attributes and methods to **outside code** (other classes or modules).
Python does **not enforce strict access control** but follows **naming conventions** to indicate intended usage.

---

### **Types of Access Modifiers**

| Modifier Type | Syntax (Prefix)        | Description                   | Access Level                                         |
| ------------- | ---------------------- | ----------------------------- | ---------------------------------------------------- |
| Public        | No underscore          | Accessible from anywhere      | Accessible everywhere                                |
| Protected     | Single underscore `_`  | Intended for internal use     | Accessible within class & subclasses (by convention) |
| Private       | Double underscore `__` | Name mangling to limit access | Accessible only inside class (name mangled)          |

---

### **Details**

---

#### **Public Members**

* No underscore prefix.
* Can be accessed from inside or outside the class freely.

```python
class MyClass:
    def __init__(self):
        self.value = 10  # public attribute

obj = MyClass()
print(obj.value)  # Accessible
```

---

#### **Protected Members**

* Prefix with a single underscore `_`.
* Conventionally treated as **internal use**; should not be accessed directly outside class or subclasses.
* No actual enforcement by Python (just a hint to developers).

```python
class MyClass:
    def __init__(self):
        self._value = 20  # protected attribute

class ChildClass(MyClass):
    def get_value(self):
        return self._value

obj = MyClass()
print(obj._value)  # Possible but discouraged
```

---

#### **Private Members**

* Prefix with double underscore `__`.
* Triggers **name mangling**: Python internally changes the name to `_ClassName__member`.
* Makes it harder (but not impossible) to access outside class.

```python
class MyClass:
    def __init__(self):
        self.__value = 30  # private attribute

    def get_value(self):
        return self.__value

obj = MyClass()
print(obj.get_value())        # Works fine
# print(obj.__value)          # AttributeError
print(obj._MyClass__value)    # Access via name mangling (not recommended)
```

---

### **Name Mangling**

* Used only with double underscore prefix.
* Helps avoid naming conflicts in subclasses.
* Access outside the class requires `_ClassName__member` syntax.

---

### **Summary Table**

| Modifier  | Prefix | Accessible From    | Python Enforcement  | Purpose                     |
| --------- | ------ | ------------------ | ------------------- | --------------------------- |
| Public    | None   | Anywhere           | No                  | General access              |
| Protected | `_`    | Class & subclasses | No (convention)     | Internal use only           |
| Private   | `__`   | Inside class only  | Yes (name mangling) | Stronger encapsulation hint |

---

### **Best Practices**

* Use public for normal attributes/methods.
* Use protected `_` for internal or subclass usage.
* Use private `__` to avoid accidental override or access, but remember it’s still accessible via name mangling if needed.
* Encapsulation in Python relies mostly on **developer discipline and conventions**.

---
