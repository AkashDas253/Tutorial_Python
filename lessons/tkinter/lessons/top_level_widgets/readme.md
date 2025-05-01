## **Top-Level Widgets in Tkinter**

In Tkinter, top-level widgets are windows or pop-up dialog boxes that are independent of the main application window. They provide a way to create additional windows that interact with the main window or perform standalone tasks.

### **🖥 Top-Level Widget: `Toplevel`**

The `Toplevel` widget in Tkinter is used to create a new, separate window from the root window. It can be used for creating secondary windows, pop-ups, or dialogs.

---

### **🔧 Syntax of `Toplevel`**

```python
toplevel = tk.Toplevel(master=None, cnf={}, **kwargs)
```

### **Parameters**

- **`master`**: The parent widget. It defaults to `None`, which means it is associated with the root window.
- **`cnf`**: Configuration options passed in the form of a dictionary (optional).
- **`kwargs`**: Other keyword arguments that configure the widget (e.g., `title`, `geometry`, etc.).

---

### **Attributes of `Toplevel` Widget**

1. **`geometry()`**: Specifies the dimensions and position of the window (e.g., `"300x200+100+100"`).
2. **`title()`**: Sets the title of the window.
3. **`resizable()`**: Controls whether the window is resizable. Takes two boolean values (width, height).
4. **`iconbitmap()`**: Sets an icon for the window using a `.ico` file.
5. **`withdraw()`**: Hides the window (does not destroy it).
6. **`deiconify()`**: Shows a hidden window.
7. **`protocol()`**: Associates a handler with specific events, such as closing the window.

---

### **Example Usage of `Toplevel`**

```python
import tkinter as tk

# Function to create a secondary window
def open_window():
    # Creating a new top-level window
    top = tk.Toplevel(root)
    top.title("Secondary Window")
    top.geometry("250x150")
    label = tk.Label(top, text="This is a secondary window")
    label.pack()

# Main root window
root = tk.Tk()
root.title("Main Window")

# Button to open secondary window
button = tk.Button(root, text="Open Window", command=open_window)
button.pack()

root.mainloop()
```

In this example:
- Clicking the "Open Window" button opens a secondary window created with `Toplevel`.
- The secondary window has a title and a label widget.

---

### **Handling Window Close Event**

You can bind specific actions to the close event of a `Toplevel` window using the `protocol()` method.

```python
import tkinter as tk

def on_close():
    print("Window is being closed!")
    top.destroy()

root = tk.Tk()
top = tk.Toplevel(root)

# Set title and geometry of the top-level window
top.title("Close Event Example")
top.geometry("200x100")

# Bind the close event of the top-level window to the on_close function
top.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()
```

- When the user closes the `Toplevel` window, the `on_close()` function is called, allowing you to define specific behavior (like saving data or confirming the close).

---

### **Common Methods for `Toplevel`**

| Method          | Description |
|-----------------|-------------|
| `geometry()`    | Set the size and position of the window. |
| `title()`       | Set the title of the window. |
| `resizable()`   | Make the window resizable (width, height). |
| `iconbitmap()`  | Set a custom icon for the window. |
| `withdraw()`    | Hide the window. |
| `deiconify()`   | Show the window. |
| `protocol()`    | Bind specific actions to events like window closing. |

---

### **Use Case Example: Modal Dialog**

A common use of `Toplevel` is creating modal dialog boxes that require interaction before returning control to the main window.

```python
import tkinter as tk

def open_modal():
    top = tk.Toplevel(root)
    top.title("Modal Dialog")
    top.geometry("200x100")

    label = tk.Label(top, text="This is a modal dialog")
    label.pack()

    button = tk.Button(top, text="Close", command=top.destroy)
    button.pack()

    # Disable the main window while the modal is open
    top.grab_set()  # Makes the top-level window modal

root = tk.Tk()
root.title("Main Window")
root.geometry("300x200")

button = tk.Button(root, text="Open Modal", command=open_modal)
button.pack()

root.mainloop()
```

In this example:
- The `top.grab_set()` method is used to disable interaction with the main window until the modal dialog is closed.

---

### **`Toplevel` vs. `Tk`**

- **`Toplevel`** is used for creating secondary windows in a Tkinter application.
- **`Tk`** is the main root window of the application.
  
Key differences:
- `Tk` is the primary window, while `Toplevel` is used for secondary windows.
- You can have multiple `Toplevel` windows, but only one `Tk` root window.

---

## **Summary of `Toplevel` Widget**

| Feature            | Description |
|--------------------|-------------|
| **Purpose**        | Create secondary windows in an application |
| **Usage**          | `Toplevel(master)` |
| **Common Methods** | `geometry()`, `title()`, `resizable()`, `iconbitmap()`, `withdraw()`, `deiconify()`, `protocol()` |

---

This note covers the essential aspects of the `Toplevel` widget in Tkinter. Let me know if you need more details or examples!