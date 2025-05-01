## Menus in PySimpleGUI

Menus in PySimpleGUI provide a traditional way to organize actions in a top-level bar within a window, similar to menus found in desktop applications. They are especially useful for organizing commands, file operations, settings, or help options.

---

### Core Concept

Menus in PySimpleGUI use the `sg.Menu()` element and are designed using nested Python lists (or tuples). Each top-level menu item can have submenus, dividers, disabled entries, or nested menus.

---

### Syntax

```python
sg.Menu(
    menu_definition,  # Nested list or tuple defining the menu
    tearoff=False,    # If True, allows menu to be detached (Tkinter-specific)
    background_color=None,
    text_color=None,
    disabled_text_color=None,
    font=None,
    pad=None,
    key=None
)
```

---

### Menu Definition Structure

The menu is defined as a list of tuples:

```python
menu_def = [
    ['File', ['Open', 'Save', '---', 'Exit']],
    ['Edit', ['Undo', ['Redo', 'Repeat'], 'Cut', 'Copy', 'Paste']],
    ['Help', ['About']]
]
```

- **'---'**: Menu divider.
- **Nested lists**: Submenus.
- **Strings**: Event keys.
- **Menu entries can be disabled** using special constructs or conditions.

---

### Basic Example

```python
import PySimpleGUI as sg

menu_def = [['File', ['Open', 'Save', '---', 'Exit']],
            ['Edit', ['Paste', ['Special', 'Normal'], 'Undo']],
            ['Help', 'About...']]

layout = [
    [sg.Menu(menu_def)],
    [sg.Text('Right-click the menu for options')]
]

window = sg.Window('Menu Example', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    sg.popup(f'You selected: {event}')

window.close()
```

---

### Event Handling

- Menu selections generate **string events** matching the menu item name.
- These can be captured in the event loop like any other GUI element.

---

### Right-Click Context Menus

You can also define context menus for right-clicking on specific elements or the window itself:

```python
right_click_menu = ['Unused', ['Copy', 'Paste', 'Delete']]

layout = [
    [sg.Text('Right-click me', right_click_menu=right_click_menu)]
]
```

---

### Special Menu Features

| Feature                | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Nested submenus        | Achieved via nested lists (recursive).                                      |
| Dividers (`'---'`)     | Visually separate groups of options.                                        |
| Disabled items         | Use `sg.MENU_DISABLED_CHARACTER` prefix (usually '::') to mark items.       |
| Keyboard shortcuts     | Add after name with `::Key` to trigger actions.                             |
| Tearoff                | If `True`, menus can be torn off into their own window (Tkinter only).      |

---

### Keyboard Shortcuts (via Custom Keys)

Although PySimpleGUI doesn't directly support platform-wide hotkeys, you can simulate shortcuts:

```python
menu_def = [['File', ['Open::open', 'Exit::exit']]]
# Later check for 'open' or 'exit' in event handling
```

---

### Summary

- Menus in PySimpleGUI are defined using nested lists of items.
- Use `sg.Menu()` for the top-level menu bar.
- Use right-click context menus for specific elements.
- Menu events are plain strings matching the item names.
- Keyboard shortcut simulation and nested submenus are supported.
