## **Scope and Access in Python**  

### **Definition**  
- **Scope** defines where a variable can be accessed within a program.  
- **Access** refers to retrieving or modifying a variable within its scope.  
- Python follows the **LEGB rule** (Local → Enclosing → Global → Built-in) to resolve variable names.  

---

### **LEGB Rule (Variable Resolution Order)**  
| Scope Level | Description | Example |
|-------------|------------|---------|
| **Local (L)** | Inside a function or block | Variables declared inside a function |
| **Enclosing (E)** | Inside a nested function | Variables from the outer (non-global) function |
| **Global (G)** | Defined at the top level of the script | Variables declared outside any function |
| **Built-in (B)** | Predefined by Python | Built-in functions like `print()`, `len()` |

```python
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # "local"
    
    inner()
    print(x)  # "enclosing"

outer()
print(x)  # "global"
```

---

### **Local Scope**  
- Variables declared inside a function are **local** to that function.  
- Cannot be accessed outside the function.  

```python
def func():
    a = 10  # Local variable
    print(a)

func()
# print(a)  # Error: NameError (a is not defined outside the function)
```

---

### **Global Scope**  
- Variables declared outside any function are **global**.  
- Accessible throughout the program.  

```python
x = 100  # Global variable

def func():
    print(x)  # Accessing global variable

func()
print(x)  # 100
```

---

### **Modifying Global Variables** (`global` keyword)  
- To modify a global variable inside a function, use `global`.  

```python
y = 5

def update():
    global y  # Allows modification of global variable
    y = 10

update()
print(y)  # 10
```

---

### **Enclosing Scope (Nonlocal Variables)**  
- **Used in nested functions** to modify outer function variables.  
- Use `nonlocal` keyword to modify an enclosing variable.  

```python
def outer():
    x = "enclosing"

    def inner():
        nonlocal x
        x = "modified"
    
    inner()
    print(x)  # "modified"

outer()
```

---

### **Built-in Scope**  
- Contains predefined functions like `print()`, `len()`, `range()`.  

```python
print(len([1, 2, 3]))  # 3 (Built-in len function)
```

---

### **Variable Lifetime**  
- **Local variables** exist only inside the function.  
- **Global variables** exist until the program terminates.  

```python
def test():
    x = 50  # Local, destroyed after function ends

test()
# print(x)  # Error: NameError
```

---
---


## Scope and Access in Python

### Scope
In Python, the scope of a variable determines the region of the code where the variable is accessible. Python uses the LEGB rule to resolve the scope of variables, which stands for Local, Enclosing, Global, and Built-in scopes.

#### LEGB Rule

1. **Local Scope**: Variables defined within a function.
2. **Enclosing Scope**: Variables in the local scope of enclosing functions (nested functions).
3. **Global Scope**: Variables defined at the top level of a module or declared global using the `global` keyword.
4. **Built-in Scope**: Names preassigned in Python (e.g., `len`, `range`).

#### Local Scope

Variables defined inside a function are in the local scope and can only be accessed within that function.

```python
def my_function():
    local_var = 10  # Local scope
    print(local_var)

my_function()  # Output: 10
# print(local_var)  # Error: NameError: name 'local_var' is not defined
```

#### Enclosing Scope

Variables in the local scope of enclosing functions are accessible in nested functions.

```python
def outer_function():
    enclosing_var = "Hello"

    def inner_function():
        print(enclosing_var)  # Enclosing scope

    inner_function()

outer_function()  # Output: Hello
```

#### Global Scope

Variables defined at the top level of a module or declared global using the `global` keyword are in the global scope.

```python
global_var = "I am global"

def my_function():
    print(global_var)  # Global scope

my_function()  # Output: I am global
print(global_var)  # Output: I am global
```

To modify a global variable inside a function, use the `global` keyword.

```python
counter = 0

def increment():
    global counter
    counter += 1

increment()
print(counter)  # Output: 1
```

#### Built-in Scope

Python has a set of built-in functions and variables that are always available.

```python
print(len("Hello"))  # Built-in scope, Output: 5
```

## Working with Scope

### Naming Variables

If you operate with the same variable name inside and outside of a function, Python will treat them as two separate variables, one available in the global scope (outside the function) and one available in the local scope (inside the function).

#### Syntax
```python
x = 300

def myfunc():
    x = 200
    print(x)

myfunc()
print(x)
```

### Global Keyword

- If you need to create a global variable but are stuck in the local scope, you can use the `global` keyword.
- The `global` keyword makes the variable global.

#### Syntax
```python
def myfunc():
    global x
    x = 300

myfunc()
print(x)
```

- Use the `global` keyword if you want to make a change to a global variable inside a function.

#### Syntax
```python
x = 300

def myfunc():
    global x
    x = 200

myfunc()
print(x)
```

### Nonlocal Keyword

- The `nonlocal` keyword is used to work with variables inside nested functions.
- The `nonlocal` keyword makes the variable belong to the outer function.

#### Syntax
```python
def myfunc1():
    x = "Jane"
    def myfunc2():
        nonlocal x
        x = "hello"
    myfunc2()
    return x

print(myfunc1())
```

## Access Modifiers

Python does not have traditional access modifiers like `public`, `protected`, and `private` found in other languages. Instead, it uses naming conventions to indicate the intended scope of variables and methods.

1. **Public**: Accessible from anywhere. By default, all members are public.
2. **Protected**: Indicated by a single underscore `_`. Intended for internal use.
3. **Private**: Indicated by a double underscore `__`. Name mangling is used to make it harder to access from outside the class.

```python
class MyClass:
    def __init__(self):
        self.public_var = "Public"
        self._protected_var = "Protected"
        self.__private_var = "Private"

    def public_method(self):
        return "Public method"

    def _protected_method(self):
        return "Protected method"

    def __private_method(self):
        return "Private method"

obj = MyClass()
print(obj.public_var)  # Output: Public
print(obj._protected_var)  # Output: Protected
# print(obj.__private_var)  # Error: AttributeError

print(obj.public_method())  # Output: Public method
print(obj._protected_method())  # Output: Protected method
# print(obj.__private_method())  # Error: AttributeError

# Accessing private variables and methods using name mangling
print(obj._MyClass__private_var)  # Output: Private
print(obj._MyClass__private_method())  # Output: Private method
```

### Summary Table

| Scope          | Description                                                                 | Syntax Example                                      |
|----------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| Local          | Variables defined within a function                                         | `def func(): local_var = 10`                        |
| Enclosing      | Variables in the local scope of enclosing functions                         | `def outer(): enclosing_var = 5; def inner(): ...`  |
| Global         | Variables defined at the top level of a module or declared global            | `global_var = 20; global global_var`                |
| Built-in       | Names preassigned in Python                                                 | `print(len("Hello"))`                               |
| Public         | Accessible from anywhere (default)                                          | `self.public_var = "Public"`                        |
| Protected      | Intended for internal use (single underscore)                               | `self._protected_var = "Protected"`                 |
| Private        | Name mangling to make it harder to access (double underscore)               | `self.__private_var = "Private"`                    |