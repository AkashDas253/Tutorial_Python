## TKinter

### 1. **Widgets**
   - **Label**: A widget to display text or images.
   - **Button**: A clickable button widget.
   - **Entry**: A single-line text entry field.
   - **Text**: A multi-line text field.
   - **Checkbutton**: A widget that allows the user to select or deselect an option.
   - **Radiobutton**: A set of options, where only one option can be selected at a time.
   - **Listbox**: A widget that displays a list of items.
   - **Scrollbar**: A scrollbar for scrolling through widgets like Text and Listbox.
   - **Canvas**: A widget for drawing shapes, images, and other graphics.
   - **Frame**: A container widget that holds other widgets.
   - **Menu**: A menu that can contain options and submenus.
   - **Toplevel**: A widget used to create new top-level windows.
   - **Spinbox**: A widget for entering a value from a given range.
   - **Scale**: A widget for selecting a numeric value from a range via a sliding bar.
   - **PanedWindow**: A container widget that allows resizing of its child widgets.
   - **OptionMenu**: A widget for selecting from a list of options.

### 2. **Geometry Management**
   - **pack()**: A method for packing widgets in the parent container.
   - **grid()**: A method to place widgets in a grid.
   - **place()**: A method to place widgets at a specific location.

### 3. **Event Handling**
   - **Binding events**: Binding a function or method to a specific event, such as a button click or key press.
   - **Event types**: Mouse, keyboard, focus, and other types of events.

### 4. **Layouts**
   - **pack() options**: `side`, `fill`, `expand`, etc.
   - **grid() options**: `row`, `column`, `sticky`, `rowspan`, `columnspan`, etc.
   - **place() options**: `x`, `y`, `relx`, `rely`, `anchor`, etc.

### 5. **Tkinter Main Loop**
   - **mainloop()**: The main event loop that runs the Tkinter application.

### 6. **Tkinter Classes and Methods**
   - **Tk**: The main Tkinter class used to create a window.
   - **Toplevel**: A class used to create additional windows.
   - **Canvas methods**: `create_line()`, `create_rectangle()`, `create_oval()`, etc.
   - **Text widget methods**: `insert()`, `delete()`, `get()`, etc.

### 7. **Color and Fonts**
   - **Colors**: Set widget colors using color names or hex codes.
   - **Fonts**: Set the font for widgets using the `font` option.

### 8. **Dialogs**
   - **MessageBox**: Used for showing messages like `showinfo()`, `showerror()`, `askyesno()`, etc.
   - **File Dialogs**: `askopenfilename()`, `asksaveasfilename()`, etc.

### 9. **Top-level Widgets**
   - **Menu**: Creating and managing menus.
   - **ToolTip**: Showing hints when hovering over widgets (can be custom).
   - **PanedWindow**: A widget that can be divided into resizable sections.
   
### 10. **Attributes**
   - **Widget configuration**: Setting options using `.config()` or during initialization.
   - **Geometry**: Setting window size and position using `.geometry()`.

### 11. **Additional Concepts**
   - **Variable types**: `StringVar()`, `IntVar()`, `DoubleVar()`, `BooleanVar()` for variable binding.
   - **Callbacks**: Functions or methods assigned to events like button clicks.
   - **Dialogs**: File dialogs, message boxes, and color pickers.
   - **Threads**: Running background tasks in a Tkinter app (though it needs careful handling due to Tkinter’s main thread restrictions).
