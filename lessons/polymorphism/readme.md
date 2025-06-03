## **Polymorphism in Python**

---

### **Definition**

**Polymorphism** means **"many forms"**. It allows the **same interface or method name** to behave **differently based on the object or context**. It enables **flexible and extensible code** in object-oriented programming.

---

### **Purpose**

* Write generic code that works with different data types or classes
* Enable interface consistency across different objects
* Improve code readability and reusability

---

### **Types of Polymorphism in Python**

| Type                 | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| Compile-time         | Not supported in Python (no method overloading based on signatures) |
| Run-time             | Supported via method overriding                                     |
| Duck Typing          | Type is determined by behavior, not inheritance                     |
| Operator Overloading | Same operator behaves differently for different types               |

---

### **Built-in Function Polymorphism**

Python built-in functions like `len()`, `type()`, `max()`, etc., work with various data types, demonstrating function polymorphism.

```python
print(len("Hello"))          # String: 5
print(len([1, 2, 3]))        # List: 3
print(len({"a": 1, "b": 2})) # Dict: 2
```

---

### **1. Method Overriding (Runtime Polymorphism)**

A subclass provides a different implementation of a method from its superclass.

```python
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):
    def speak(self):
        print("Dog barks")

a = Animal()
d = Dog()

a.speak()  # Animal speaks
d.speak()  # Dog barks
```

---

### **2. Duck Typing**

Python focuses on **behavior, not type**: “If it looks like a duck and quacks like a duck, it’s a duck.”

```python
class Cat:
    def speak(self):
        print("Meow")

class Human:
    def speak(self):
        print("Hello")

def make_it_speak(entity):
    entity.speak()

make_it_speak(Cat())
make_it_speak(Human())
```

---

### **Class-based Polymorphism Without Inheritance**

Even if classes do not inherit from a common base class, polymorphism works when they define methods with the same name.

```python
class Car:
    def move(self):
        print("Drive")

class Boat:
    def move(self):
        print("Sail")

class Plane:
    def move(self):
        print("Fly")

for vehicle in (Car(), Boat(), Plane()):
    vehicle.move()
```

---

### **3. Operator Overloading**

Python allows **special methods** to redefine operators for custom objects.

| Operator | Special Method         |
| -------- | ---------------------- |
| `+`      | `__add__(self, other)` |
| `-`      | `__sub__`              |
| `*`      | `__mul__`              |
| `/`      | `__truediv__`          |

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v1 = Vector(1, 2)
v2 = Vector(3, 4)
v3 = v1 + v2  # Calls v1.__add__(v2)
```

---

### **Polymorphism with Functions and Objects**

```python
class Bird:
    def intro(self):
        print("I'm a bird")
    def fly(self):
        print("I fly in the sky")

class Sparrow(Bird):
    def fly(self):
        print("I fly short distances")

class Eagle(Bird):
    def fly(self):
        print("I soar high")

def bird_flight(bird):
    bird.fly()

bird_flight(Sparrow())
bird_flight(Eagle())
```

---

### **Benefits**

* Enhances code flexibility and scalability
* Enables interface reuse
* Supports dynamic behavior without modifying existing code

---
