## Python’s `pdb` Module

The `pdb` (Python Debugger) module is Python’s built-in interactive source code debugger. It allows developers to set breakpoints, inspect variables, control execution flow, and evaluate expressions in real time.

---

### Overview

* Built-in (no installation needed).
* Supports **line-by-line** execution control.
* Works both **inside scripts** and **from the command line**.
* Can be used programmatically or interactively.

---

### Starting the Debugger

#### 1. **From Command Line**

```bash
python -m pdb script.py
```

* Runs the script under the debugger from the start.

#### 2. **Inside Code**

```python
import pdb
pdb.set_trace()  # Execution stops here
```

* Drops into an interactive debugging session at that point.

#### 3. **Post-Mortem Debugging** (after an exception)

```bash
python -m pdb script.py
```

or programmatically:

```python
import pdb
pdb.post_mortem()
```

---

### Common Commands in `pdb`

| Command           | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `h` / `help`      | Show help.                                                      |
| `h <command>`     | Show help for a specific command.                               |
| `l` / `list`      | List source code around the current line.                       |
| `n` / `next`      | Execute the next line (step over function calls).               |
| `s` / `step`      | Step into the current function call.                            |
| `c` / `continue`  | Continue execution until the next breakpoint.                   |
| `b <line>`        | Set a breakpoint at a specific line number in the current file. |
| `b <file>:<line>` | Set a breakpoint in another file.                               |
| `b <function>`    | Set a breakpoint at the start of a function.                    |
| `cl` / `clear`    | Clear all breakpoints or a specific one.                        |
| `p <expr>`        | Print the value of an expression.                               |
| `pp <expr>`       | Pretty-print the value of an expression.                        |
| `whatis <expr>`   | Show the type of an expression.                                 |
| `where` / `w`     | Show the call stack.                                            |
| `u` / `up`        | Move up one frame in the call stack.                            |
| `d` / `down`      | Move down one frame in the call stack.                          |
| `q` / `quit`      | Exit the debugger.                                              |
| `!<python>`       | Execute a Python statement in the current context.              |

---

### Breakpoint Management

* Set multiple breakpoints:

  ```pdb
  b 12
  b mymodule.py:45
  b my_function
  ```
* List all breakpoints:

  ```pdb
  b
  ```
* Clear a specific breakpoint:

  ```pdb
  cl 2
  ```

---

### Inspecting State

* `p variable` – Prints a variable value.
* `pp variable` – Pretty prints a variable.
* `dir(object)` – Lists object attributes.
* `locals()` – Shows all local variables.
* `globals()` – Shows global variables.

---

### Running to a Specific Location

* `until` – Run until a line greater than the current one in the same function.
* `j <line>` – Jump execution to a new line number (dangerous; skips code).

---

### Advanced Usage

* **Conditional Breakpoints**:

  ```pdb
  b 20, x > 10
  ```
* **Ignore Breakpoint Count**:

  ```pdb
  b 20, ignore 3
  ```
* **Run Commands Automatically**:

  ```bash
  python -m pdb -c "b 10" -c "c" script.py
  ```
* **Post-Mortem from Traceback**:

  ```python
  import pdb, sys
  pdb.post_mortem(sys.last_traceback)
  ```

---

### Usage Scenarios

* Step-by-step debugging during development.
* Diagnosing issues in production code (via post-mortem).
* Inspecting variable state at a specific execution point.
* Debugging complex recursive or asynchronous logic.

---
