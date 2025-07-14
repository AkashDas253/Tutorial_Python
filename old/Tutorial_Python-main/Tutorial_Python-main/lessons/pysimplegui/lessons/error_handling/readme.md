## Error Handling in PySimpleGUI

Error handling in PySimpleGUI is about detecting, preventing, and managing exceptions or unexpected behaviors during GUI creation and event handling. Since PySimpleGUI is a wrapper around other GUI frameworks (Tkinter, Qt, WxPython, Web), it inherits many runtime behaviors from them.

---

### Goals of Error Handling

- Prevent crashes from user input or GUI misuse  
- Provide user-friendly feedback  
- Catch exceptions from function calls or element misuse  
- Validate data from inputs before processing  

---

### Common Error Types

| Type                       | Description                                           |
|----------------------------|-------------------------------------------------------|
| Element misuse             | Accessing or updating non-existent keys              |
| Invalid layout             | Malformed layout lists or elements                   |
| Input conversion errors    | Wrong type conversions (e.g., str to int)            |
| Window reuse errors        | Reusing closed or destroyed window                   |
| Thread safety issues       | Updating GUI from background threads (requires care) |

---

### Exception Handling Pattern

Wrap the main loop in a `try-except` block to catch and respond to unexpected exceptions:

```python
try:
    window = sg.Window('Demo', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        # Your logic here
except Exception as e:
    sg.popup_error(f"An error occurred:\n{e}")
finally:
    window.close()
```

---

### Input Validation Example

```python
event, values = window.read()
if event == 'Submit':
    try:
        num = int(values['-NUM-'])
    except ValueError:
        sg.popup_error('Please enter a valid number.')
```

---

### Handling Missing Keys or Elements

Always check key existence before updating elements:

```python
if '-OUTPUT-' in window.AllKeysDict:
    window['-OUTPUT-'].update('Result')
else:
    sg.popup_error('Output key not found')
```

---

### Defensive Programming Practices

- Use default values when fetching inputs  
- Validate all user inputs before processing  
- Avoid logic that assumes element availability  
- Use `update()` carefully after checking for key existence  

---

### PySimpleGUI Debug Features

- `sg.Print()`: Sends output to a separate debug window  
- `sg.popup_error()`: Displays error messages in a modal popup  
- `sg.set_options(debug_win_size=(width, height))`: Controls debug window  

---

### Thread Safety

PySimpleGUI is **not thread-safe**. All GUI updates must be done from the **main thread**. Use `window.write_event_value()` to send safe messages from threads:

```python
def background_task():
    # Do work...
    window.write_event_value('-THREAD-', result)
```

---

### Summary

- Wrap GUI logic in try-except blocks  
- Use `popup_error()` for user feedback  
- Validate and sanitize inputs  
- Handle missing keys or layout errors  
- Use `write_event_value()` for thread-safe updates  
