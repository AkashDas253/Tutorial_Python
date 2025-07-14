## **Metaclass in Python**

---

### What is a Metaclass?

A **metaclass** in Python is a "class of a class"—it defines how classes themselves behave. In Python, *everything is an object*, including classes, and **metaclasses define the behavior of these classes**, just as classes define the behavior of instances.

---

### Key Concepts

| Term                    | Description                                                |
| ----------------------- | ---------------------------------------------------------- |
| **Class**               | Defines behavior of objects (instances).                   |
| **Metaclass**           | Defines behavior of classes.                               |
| **`type`**              | Built-in metaclass; the default metaclass for all classes. |
| **Custom Metaclass**    | Subclass of `type`, allowing control over class creation.  |
| **`__metaclass__`**     | Legacy way (Python 2) to define a metaclass.               |
| **`metaclass` keyword** | Used in Python 3 to assign a metaclass.                    |

---

### Why Use a Metaclass?

* To enforce coding standards (e.g., class must have certain attributes).
* To register classes automatically.
* To inject methods/attributes into classes.
* To modify class hierarchy or structure at creation time.

---

### How Class Creation Works Internally

1. **Define class** → Class body is executed in a namespace (dict).
2. **Metaclass is called** with:

   * `name`: name of class being defined
   * `bases`: base classes
   * `namespace`: attributes/methods
3. Metaclass returns the final class object.

---

### Default Behavior with `type`

```python
# Equivalent to: class MyClass: pass
MyClass = type('MyClass', (), {})
```

This dynamically creates a class using the default metaclass `type`.

---

### Defining a Custom Metaclass

```python
class MyMeta(type):
    def __new__(cls, name, bases, dct):
        print(f'Creating class {name}')
        dct['created_by_metaclass'] = True
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MyMeta):
    pass

print(MyClass.created_by_metaclass)  # True
```

---

### Lifecycle Methods in Metaclass

| Method                            | Purpose                                                         |
| --------------------------------- | --------------------------------------------------------------- |
| `__new__(mcs, name, bases, dct)`  | Controls the creation of the class.                             |
| `__init__(cls, name, bases, dct)` | Initializes the class after it is created.                      |
| `__call__(cls, *args, **kwargs)`  | Controls instantiation of the class itself (rarely overridden). |

---

### Example: Enforcing Rules

```python
class InterfaceChecker(type):
    def __init__(cls, name, bases, dct):
        if not hasattr(cls, 'process'):
            raise TypeError("Classes must define a 'process' method")

class MyValidClass(metaclass=InterfaceChecker):
    def process(self):
        pass

# class MyInvalidClass(metaclass=InterfaceChecker):
#     pass  # Raises TypeError
```

---

### Use Cases

| Use Case             | Description                                                          |
| -------------------- | -------------------------------------------------------------------- |
| **ORMs**             | Automatically mapping classes to database tables (e.g., SQLAlchemy). |
| **Frameworks**       | Enforce design patterns or registration (e.g., Django models).       |
| **Code Enforcement** | Enforce structure like interfaces.                                   |
| **Plugin Systems**   | Automatically registering plugins.                                   |

---

### Comparison: Class vs Metaclass

| Feature      | Class                 | Metaclass             |
| ------------ | --------------------- | --------------------- |
| Operates on  | Instances             | Classes               |
| Defined by   | `class` keyword       | Subclassing `type`    |
| Custom logic | `__init__`, `__new__` | `__init__`, `__new__` |
| Keyword used | None                  | `metaclass=` keyword  |

---

### Best Practices

* Avoid metaclasses unless absolutely necessary.
* Use class decorators for simpler use cases.
* Use them when working with **class-level logic**, not instance-level logic.

---

### Related Concepts

| Concept              | Description                                                                 |
| -------------------- | --------------------------------------------------------------------------- |
| **Class Decorators** | Modify or enhance a class without creating a metaclass.                     |
| **Descriptors**      | Control attribute access on instances.                                      |
| **`__slots__`**      | Restrict dynamic creation of attributes (unrelated but sometimes confused). |

---

### Summary

Metaclasses allow **control over class creation**, just as classes control **instance creation**. They're powerful but should be used carefully due to complexity. They're most useful in **framework design, code enforcement, and automation of class behaviors**.
