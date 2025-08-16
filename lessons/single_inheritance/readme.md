# Single Inheritance in Python 

**Inheritance** is an OOP concept where one class (**child/derived**) can access the properties and methods of another class (**parent/base**).

**Single inheritance** means a class **inherits from exactly one parent class**.

---

## Syntax

```python
class Parent:
    # parent attributes and methods
    pass

class Child(Parent):
    # child inherits everything from Parent
    pass
```

---

## Key Features

* Only **one parent → one child** relationship.
* Child inherits **all non-private attributes and methods** of parent.
* Child can **add new methods/attributes**.
* Child can **override parent methods**.
* Parent methods can still be accessed using **`super()`** or `Parent.method(self, ...)`.

---

## Example – Basic Single Inheritance

```python
class Animal:
    def speak(self):
        print("Animal makes a sound")

class Dog(Animal):  # Single inheritance
    def bark(self):
        print("Dog barks")

d = Dog()
d.speak()  # Inherited
d.bark()   # Child-specific
```

---

## Constructor in Single Inheritance

Child can override the parent constructor. Use `super()` to call parent’s constructor.

```python
class Person:
    def __init__(self, name):
        self.name = name
        print("Person initialized")

class Student(Person):
    def __init__(self, name, roll):
        super().__init__(name)   # Call parent constructor
        self.roll = roll
        print("Student initialized")

s = Student("Alice", 101)
```

---

## Method Overriding in Single Inheritance

Child can redefine parent’s method.

```python
class Parent:
    def greet(self):
        print("Hello from Parent")

class Child(Parent):
    def greet(self):  # Override
        print("Hello from Child")

c = Child()
c.greet()  # Child’s version
```

---

## Attribute Overriding

Child can override parent attributes.

```python
class Parent:
    value = 10

class Child(Parent):
    value = 20  # Overrides parent

print(Child().value)  # 20
```

---

## Accessing Parent Methods

Using `super()`:

```python
class Parent:
    def show(self):
        print("Parent show()")

class Child(Parent):
    def show(self):
        super().show()  # Call parent method
        print("Child show()")

Child().show()
```

---

## Advantages of Single Inheritance

* **Code Reusability** → Child reuses parent’s code.
* **Maintainability** → Changes in parent propagate to child.
* **Extensibility** → Child can extend parent’s functionality.
* **Simplicity** → Easier than multiple inheritance (no ambiguity).

---

## Limitations

* Child is limited to **only one parent**.
* If functionality is spread across multiple unrelated classes, **multiple inheritance** may be needed.

---

## Summary

* **Single Inheritance** = one parent, one child.
* Child inherits parent’s **methods and attributes**.
* Supports **constructor overriding**, **method overriding**, and **attribute overriding**.
* `super()` allows calling parent’s methods/constructors.
* Used for **clear, simple, and reusable class hierarchies**.

---
