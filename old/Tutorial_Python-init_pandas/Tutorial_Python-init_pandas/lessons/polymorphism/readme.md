## Python Polymorphism

### Definition

- Polymorphism means "many forms".
- In programming, it refers to methods/functions/operators with the same name that can be executed on many objects or classes.

### Function Polymorphism

- Example of a Python function that can be used on different objects: `len()` function.

#### String

- For strings, `len()` returns the number of characters.

#### Syntax
```python
x = "Hello World```"
print(len(x))
```

#### Tuple

- For tuples, `len()` returns the number of items in the tuple.

#### Syntax
```python
mytuple = ("apple", "banana", "cherry")
print(len(mytuple))
```

#### Dictionary

- For dictionaries, `len()` returns the number of key/value pairs in the dictionary.

#### Syntax
```python
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(len(thisdict))
```

### Class Polymorphism

- Polymorphism is often used in class methods, where multiple classes have the same method name.

#### Example

- Different classes with the same method:

#### Syntax
```python
class Car:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Drive```")

class Boat:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Sail```")

class Plane:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  x.move()
```

- The `for` loop at the end demonstrates polymorphism by executing the same method for all three classes.

### Inheritance Class Polymorphism

- Child classes with the same name can use polymorphism.

#### Example

- Create a parent class `Vehicle` and make `Car`, `Boat`, `Plane` child classes of `Vehicle`.

#### Syntax
```python
class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move```")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail```")

class Plane(Vehicle):
  def move(self):
    print("Fly```")

car1 = Car("Ford", "Mustang")
boat1 = Boat("Ibiza", "Touring 20")
plane1 = Plane("Boeing", "747")

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()
```

- Child classes inherit properties and methods from the parent class.
- The `Car` class is empty but inherits `brand`, `model`, and `move()` from `Vehicle`.
- The `Boat` and `Plane` classes override the `move()` method.
- Polymorphism allows executing the same method for all classes.



