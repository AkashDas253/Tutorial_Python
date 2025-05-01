## Design Patterns in PySimpleGUI

PySimpleGUI encourages simplicity and procedural design, but it also allows the use of several well-known design patterns from software engineering to make GUI applications more modular, scalable, and maintainable.

---

### Procedural Event Loop Pattern

PySimpleGUI primarily uses a **procedural event loop** rather than an object-oriented widget callback model.

- The `window.read()` call blocks and waits for events.
- Events and values are processed sequentially in the loop.

**Advantages**:
- Easier to understand and debug.
- Centralized logic flow.
- Avoids callback hell common in other GUI frameworks.

---

### Model-View-Controller (MVC)

While PySimpleGUI is not MVC by default, it supports a **loose separation** of:

- **Model**: Business logic and data (can be defined in plain functions or classes).
- **View**: The GUI layout created with `sg.Window` and elements.
- **Controller**: The event loop that reads inputs, updates views, and invokes model functions.

**Usage**:
- Define your model and data-processing functions separately.
- Use the controller (event loop) to call model logic and update views.

---

### State Machine Pattern

When your GUI requires multiple modes or states (e.g., different screens), a **state machine** pattern is helpful.

- Use a variable like `state = "MENU"` to track the current state.
- Update layout and behavior based on the state.
- Switch states inside the event loop based on user actions.

**Example**:
```python
if state == "MENU":
    # Display menu layout
elif state == "GAME":
    # Display game layout
```

---

### Observer Pattern (Manual)

Although PySimpleGUI doesn't have built-in reactive capabilities, you can **manually implement the observer pattern**:

- Maintain a data model (e.g., a dictionary or class).
- When data changes, manually call `update()` on the GUI elements subscribed to that data.

**Usage**:
- Useful for dashboard UIs that reflect real-time data.
- Combine with `threading` and `write_event_value()` for reactive updates.

---

### Template Pattern

You can define reusable layout templates for similar sections of the GUI:

```python
def create_user_row(name):
    return [sg.Text(name), sg.Input(key=f'-INPUT-{name}-')]
```

- Promotes reuse and layout consistency.
- Useful in dynamic layouts.

---

### Singleton Pattern (Optional)

You can apply the singleton pattern to make sure only one main window is active at a time.

```python
class AppWindow:
    _instance = None

    @staticmethod
    def get_instance():
        if AppWindow._instance is None:
            AppWindow._instance = sg.Window('App', layout)
        return AppWindow._instance
```

- Helps in larger apps with multiple modules needing access to a single GUI window.

---

### Command Pattern (Custom Actions)

You can use the command pattern to map GUI actions to callable functions:

```python
actions = {
    'Start': start_game,
    'Exit': exit_app
}

if event in actions:
    actions[event]()
```

- Simplifies event handling.
- Avoids long `if-elif` chains.
- Makes code more scalable.

---

### Builder Pattern (Dynamic Layouts)

Useful when building GUIs dynamically (e.g., based on config files or user input).

- Construct elements step-by-step.
- Chain layout-generating functions or use loops.

```python
layout = []
for field in config_fields:
    layout.append([sg.Text(field), sg.Input(key=field)])
```

---

### Summary of Useful Patterns

| Pattern             | Use Case                                         |
|---------------------|--------------------------------------------------|
| Procedural Loop     | Basic GUIs with sequential logic                |
| MVC (manual)        | Separation of concerns                          |
| State Machine       | Multi-screen or modal applications              |
| Observer (manual)   | Updating GUI from changing data sources         |
| Template            | Reusable layout components                      |
| Singleton           | Single GUI instance enforcement                 |
| Command             | Mapping events to action functions              |
| Builder             | Generate layouts programmatically               |

---
