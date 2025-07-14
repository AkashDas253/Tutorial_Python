## Scrollable Layouts in PySimpleGUI

Scrollable layouts in PySimpleGUI allow you to display content larger than the visible window area using scrollbars. This is essential for long forms, dynamic content, or small screens.

---

### Purpose

- Display more elements than fit in the window  
- Dynamically accommodate varying content lengths  
- Enhance usability without cluttering the interface

---

## Creating a Scrollable Layout

### Method 1: `Window` scrollable parameter

Use `layout` directly with `Window(..., scrollable=True)` to make the entire window scrollable.

```python
layout = [[sg.Text(f"Line {i}")] for i in range(50)]

window = sg.Window("Scrollable Window", layout, scrollable=True, resizable=True)
```

### Method 2: Scrollable Column Inside Window

This is the most flexible and commonly used method.

```python
scrollable_column = sg.Column(
    [[sg.Text(f"Item {i}")] for i in range(100)],
    size=(300, 400),
    scrollable=True,
    vertical_scroll_only=True
)

layout = [[scrollable_column]]
window = sg.Window("Scrollable Column", layout, resizable=True)
```

---

## Key Parameters

| Parameter             | Description                                      |
|-----------------------|--------------------------------------------------|
| `scrollable=True`     | Enables scrollbars for the element               |
| `size=(w, h)`         | Controls visible size; triggers scrollbars       |
| `vertical_scroll_only`| Adds only vertical scrollbar                     |
| `expand_x` / `expand_y` | Makes columns auto-expand                      |

---

## Scrollbar Behavior

- Scrollbars appear **only if content exceeds visible size**
- Always use a fixed `size` for the scrollable column
- `scrollable=True` has no effect if size isn't smaller than content

---

## Nested Scrollable Areas

You can place multiple scrollable columns in tabs or layouts:

```python
tab1 = sg.Tab("Tab 1", [[
    sg.Column([[sg.Text(f"Row {i}")] for i in range(100)], size=(300, 200), scrollable=True)
]])

layout = [[sg.TabGroup([[tab1]])]]
```

---

## Limitations

- Scrollbar styling is platform-dependent (based on underlying GUI framework)
- Scrollable elements do **not support mouse wheel scrolling on some platforms** (especially Linux without additional settings)
- Dynamic resizing of content within scrollable columns can be tricky without `expand_x` / `expand_y`

---

## Best Practices

- Always define a `size` for scrollable containers
- Prefer `sg.Column(..., scrollable=True)` for local scroll areas
- Combine with `sg.Frame` or `sg.Tab` for better structure
- Test scrollability on target platforms (especially if using touch or older Linux GUIs)

---
