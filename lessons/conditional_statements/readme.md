## Conditional Statements in Python

### Overview

Conditional statements control the flow of execution based on conditions. They evaluate Boolean expressions and decide which code block to execute.

---

### Types of Conditional Statements

* **`if` Statement**
  Executes a block of code if a condition is `True`.

* **`if-else` Statement**
  Executes one block if the condition is `True`, otherwise another block.

* **`if-elif-else` Statement**
  Checks multiple conditions in sequence.

* **Nested Conditional Statements**
  `if` statements placed inside other `if` statements.

* **Conditional Expressions (Ternary Operator)**
  Short form of `if-else` for inline assignments.

---

### Syntax & Examples

#### `if`

```python
x = 10
if x > 5:
    print("x is greater than 5")
```

#### `if-else`

```python
x = 3
if x > 5:
    print("x is greater than 5")
else:
    print("x is not greater than 5")
```

#### `if-elif-else`

```python
x = 5
if x > 5:
    print("x is greater than 5")
elif x == 5:
    print("x equals 5")
else:
    print("x is less than 5")
```

#### Nested `if`

```python
x = 10
if x > 5:
    if x < 15:
        print("x is between 5 and 15")
```

#### Conditional Expression (Ternary)

```python
x = 7
result = "Even" if x % 2 == 0 else "Odd"
print(result)
```

---

### Usage Scenarios

* Decision-making in programs
* Handling multiple possible outcomes
* Executing different logic based on input values
* Simplifying code with ternary operators for single-line conditions

---
