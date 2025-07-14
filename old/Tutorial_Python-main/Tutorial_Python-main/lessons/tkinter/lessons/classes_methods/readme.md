# **Tkinter Classes and Methods**

Tkinter is Python’s standard GUI (Graphical User Interface) library. It is built on top of Tcl/Tk and provides object-oriented classes and methods for GUI development.

---

## **Core Classes in Tkinter**

| Class | Description |
|-------|-------------|
| `Tk` | Main window class for the application. |
| `Toplevel` | Creates a new window (child of `Tk`). |
| `Frame` | A container for grouping widgets. |
| `Label` | Displays text/images. |
| `Button` | A clickable button. |
| `Entry` | Single-line text input field. |
| `Text` | Multi-line text input. |
| `Canvas` | Drawing area for graphics and shapes. |
| `Checkbutton` | Checkbox toggle. |
| `Radiobutton` | Radio selection options. |
| `Scale` | Slider to select numeric values. |
| `Listbox` | Displays a list of options. |
| `Menu` | Menu bar and dropdowns. |
| `Scrollbar` | Scrollbars for other widgets. |
| `Spinbox` | Numeric entry with up/down arrows. |
| `PanedWindow` | Resizable panes. |
| `Message` | Multiline text with formatting. |

---

## **`Tk` Class**

### **Description**:  
Root window of any Tkinter application. It must be instantiated before creating other widgets.

### **Syntax**
```python
root = tk.Tk()
```

### **Common Methods**
| Method | Description |
|--------|-------------|
| `mainloop()` | Starts the GUI event loop. |
| `title(str)` | Sets window title. |
| `geometry("WxH")` | Sets window size. |
| `resizable(w, h)` | Enables/disables window resizing. |
| `iconbitmap(path)` | Sets window icon. |
| `configure(bg=color)` | Sets background color. |
| `quit()` | Exits the GUI loop. |
| `destroy()` | Destroys the window and all widgets. |

---

## **`Widget` Class (Super Class)**

### **All Widgets Inherit from `Widget` Class**  
It provides universal methods for layout, configuration, and event handling.

### **Common Widget Methods**
| Method | Description |
|--------|-------------|
| `pack()`, `grid()`, `place()` | Geometry managers. |
| `config(**options)` | Set multiple attributes. |
| `cget(option)` | Get current value of an option. |
| `bind(event, handler)` | Attach event handler. |
| `destroy()` | Remove the widget. |
| `winfo_*()` | Get widget information (size, position, parent, etc.). |

---

## **Widget-Specific Methods and Options**

### **Example: `Label` Widget**

```python
label = tk.Label(master, text="Hello", bg="yellow", font=("Arial", 12))
```

| Option | Description |
|--------|-------------|
| `text` | Text to display. |
| `font` | Font family, size, and style. |
| `bg`, `fg` | Background and foreground colors. |
| `width`, `height` | Dimensions of label. |
| `image` | Display image instead of text. |
| `justify` | Text alignment (`left`, `right`, `center`). |

### **Common Methods**
| Method | Description |
|--------|-------------|
| `config()` | Modify properties. |
| `cget("text")` | Get current text. |
| `after(ms, func)` | Run a function after a delay. |

---

## **Widget Layout Methods**

These are inherited from `Widget` class:

### `pack(**options)`
| Option | Description |
|--------|-------------|
| `side` | Top, bottom, left, right. |
| `fill` | X, Y, or both. |
| `expand` | Expand to fill space. |

### `grid(**options)`
| Option | Description |
|--------|-------------|
| `row`, `column` | Row/column placement. |
| `rowspan`, `columnspan` | Span multiple cells. |
| `sticky` | Alignment in cell. |

### `place(**options)`
| Option | Description |
|--------|-------------|
| `x`, `y` | Absolute position. |
| `relx`, `rely` | Relative position (0–1). |

---

## **Example Combining Widgets and Classes**

```python
import tkinter as tk

class MyApp:
    def __init__(self, root):
        self.label = tk.Label(root, text="Welcome!", font=("Arial", 14))
        self.label.pack(pady=10)

        self.button = tk.Button(root, text="Click Me", command=self.on_click)
        self.button.pack()

    def on_click(self):
        self.label.config(text="Button Clicked!")

root = tk.Tk()
root.geometry("300x150")
app = MyApp(root)
root.mainloop()
```

---

## **Additional Widget-Related Classes**

| Class | Purpose |
|-------|---------|
| `Font` (from `tkinter.font`) | Define and reuse font styles. |
| `PhotoImage` | Handle image loading (GIF/PNG). |
| `StringVar`, `IntVar`, etc. | Link variables to widget states. |

---

## **Summary**

| Area | Usage |
|------|-------|
| `Tk()` | Starts the main window. |
| `mainloop()` | Starts event loop. |
| Widget Classes | Used to create GUI elements. |
| `config()`, `cget()` | Modify or retrieve widget properties. |
| Geometry Managers | Position widgets (`pack`, `grid`, `place`). |
| Event Methods | Bind events to functions. |

---
