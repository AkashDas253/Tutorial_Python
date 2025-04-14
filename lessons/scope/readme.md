
## Scope and Access in Python

---

### **Definition**  
- **Scope** defines where a variable can be accessed within a program.  
- **Access** refers to retrieving or modifying a variable within its scope.  
- Python follows the **LEGB rule** (Local → Enclosing → Global → Built-in) to resolve variable names.  

---

### **LEGB Rule (Name Resolution)**

| Scope Type   | Description |
|--------------|-------------|
| **Local (L)** | Variables defined inside a function. |
| **Enclosing (E)** | Variables in the local scope of enclosing functions (for nested functions). |
| **Global (G)** | Variables defined at the top-level of a module or declared `global`. |
| **Built-in (B)** | Names preassigned in Python, like `len()`, `range()`, etc. |

> Python checks in the order **Local → Enclosing → Global → Built-in** to resolve a variable name.

---

### **Variable Scope Levels**

| Type | Defined In | Accessible In | Notes |
|------|------------|----------------|-------|
| **Local** | Inside a function | Only within that function | Created when function is called |
| **Enclosing** | Outer function | Inner nested function | Useful in closures |
| **Global** | Outside all functions | Accessible anywhere (read-only inside functions unless declared `global`) | Shared across functions |
| **Built-in** | Python interpreter | Everywhere | Comes from `builtins` module |

---

### **Scope Keywords**

#### `global`

Used inside a function to indicate a variable refers to the global scope.

```python
x = 5
def update():
    global x
    x = 10  # modifies global x
```

#### `nonlocal`

Used inside nested functions to refer to variables in the **enclosing (non-global)** scope.

```python
def outer():
    x = 5
    def inner():
        nonlocal x
        x = 10  # modifies x from outer()
```

---

### **Access Modifiers (Pythonic Way)**

Python does not have strict access modifiers like `private` or `protected`, but uses **naming conventions**:

| Modifier | Syntax | Meaning |
|----------|--------|---------|
| **Public** | `var` | Accessible from anywhere |
| **Protected** | `_var` | Convention: treat as protected (internal use only) |
| **Private** | `__var` | Name mangled to prevent access outside class (`_ClassName__var`) |

---

### **Namespace vs Scope**

| Concept | Description |
|--------|-------------|
| **Namespace** | A mapping from names to objects (e.g., variable name to its value) |
| **Scope** | The textual region where a namespace is directly accessible |

Each scope (local, global, etc.) corresponds to a namespace.

---

### **Use Cases Summary**

| Use Case | Scope Needed | Keyword |
|----------|--------------|---------|
| Modify a global variable in a function | Global | `global` |
| Modify an enclosing variable from a nested function | Enclosing | `nonlocal` |
| Protect class internals from outside access | Protected/Private | `_` or `__` |

---
