## **Class Methods**

---

#### **Definition**

* Methods that receive the class itself as the first argument (`cls`).
* Can access and modify **class state** that applies across all instances.
* Decorated with `@classmethod`.

---

#### **Syntax**

```python
class MyClass:
    @classmethod
    def class_method(cls, args):
        pass
```

---

#### **Example**

```python
class Person:
    population = 0

    def __init__(self, name):
        self.name = name
        Person.population += 1

    @classmethod
    def get_population(cls):
        return cls.population

print(Person.get_population())  # 0
p1 = Person("Alice")
p2 = Person("Bob")
print(Person.get_population())  # 2
```

---
