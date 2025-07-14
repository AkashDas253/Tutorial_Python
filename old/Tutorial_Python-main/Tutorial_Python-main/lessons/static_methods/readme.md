## **Static Methods**

---

#### **Definition**

* Methods that don’t receive an automatic first argument (`self` or `cls`).
* Behave like regular functions but belong to the class’s namespace.
* Cannot access or modify instance or class state.
* Decorated with `@staticmethod`.

---

#### **Syntax**

```python
class MyClass:
    @staticmethod
    def static_method(args):
        pass
```

---

#### **Example**

```python
class Math:
    @staticmethod
    def add(x, y):
        return x + y

print(Math.add(5, 3))  # 8
```

---