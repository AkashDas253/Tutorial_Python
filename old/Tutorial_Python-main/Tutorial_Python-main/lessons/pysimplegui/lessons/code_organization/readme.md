## Code Organization in PySimpleGUI

Organizing code in a PySimpleGUI application improves readability, maintainability, and scalability. While small GUIs may work in a single script, larger applications benefit from structured code design.

---

### Goals of Code Organization

- Separate GUI layout from logic  
- Encapsulate related operations  
- Enable reuse and testing  
- Allow scaling with multiple windows or features  

---

### Recommended File Structure

```
project/
│
├── main.py               # Entry point
├── layout.py             # Layout creation functions
├── controller.py         # Event loop and interactions
├── model.py              # Business logic / data handling
├── utils.py              # Helper functions
└── config.py             # Constants and configuration
```

---

### 1. Layout Module

Encapsulate layout creation into functions:

```python
# layout.py
import PySimpleGUI as sg

def main_layout():
    return [
        [sg.Text("Name"), sg.Input(key="-NAME-")],
        [sg.Button("Submit"), sg.Button("Exit")]
    ]
```

---

### 2. Model Module

Business logic, input validation, or calculations:

```python
# model.py
def validate_name(name):
    return len(name.strip()) > 0
```

---

### 3. Controller Module

Handles the event loop:

```python
# controller.py
import PySimpleGUI as sg
from layout import main_layout
from model import validate_name

def run_main_window():
    window = sg.Window("App", main_layout())
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        elif event == "Submit":
            if validate_name(values["-NAME-"]):
                sg.popup("Valid name")
            else:
                sg.popup_error("Invalid input")
    
    window.close()
```

---

### 4. Main Entry Point

```python
# main.py
from controller import run_main_window

if __name__ == "__main__":
    run_main_window()
```

---

### Best Practices

- **Avoid global variables**. Use functions and parameters.
- **Use descriptive keys** like `-USERNAME-`, `-OUTPUT-`, not just `'name'`.
- **Keep layout static unless needed**; if dynamic, build via templates.
- **Modularize recurring GUI sections**.
- **Encapsulate thread logic** into functions or classes.

---

### Optional Enhancements

- Use object-oriented wrappers for windows (if needed).
- Use enums/constants for event names.
- Support configuration via JSON/YAML files.
- Implement logging (e.g., via Python's `logging` module).

---

### Summary

| Component      | Role                                      |
|----------------|-------------------------------------------|
| `main.py`      | Launch point                              |
| `layout.py`    | Creates GUI structure                     |
| `model.py`     | Business rules, data validation           |
| `controller.py`| Handles GUI events                        |
| `utils.py`     | Generic helpers                           |
| `config.py`    | Configurable constants                    |

---
