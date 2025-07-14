## PySimpleGUI Concepts

### Core Concepts
- Event-Driven Programming
- Window Read Loop
- Layouts as Nested Lists
- Persistent Elements
- Return Values through Events

### Windows
- `Window` Class
  - Creation
  - `read()` Method
  - `close()` Method
  - `refresh()` Method
  - `finalize()` Method
  - `perform_long_operation()` for threading

### Layout
- List of Rows
- Each Row as List of Elements
- Separators between Sections
- Columns inside Layouts
- Frames inside Layouts
- Tabs and TabGroups

### Elements (Widgets)
- Text
- Input
- Button
- Multiline
- Output
- Checkbox
- Radio
- Combo (Dropdown)
- Listbox
- Slider
- Spin
- ProgressBar
- Image
- Graph
- Table
- Tree
- Pane
- Canvas
- ButtonMenu
- MenuBar
- Column
- Frame
- Tab
- TabGroup
- CalendarButton
- ColorChooserButton
- FileBrowse, FilesBrowse
- FolderBrowse
- SaveAs
- Popup Buttons (e.g., PopupGetText)

### Element Attributes
- `key`
- `size`
- `tooltip`
- `default_value`
- `disabled`
- `visible`
- `background_color`
- `text_color`
- `font`
- `justification`
- `pad`
- `expand_x`, `expand_y`

### Element Methods
- `update()`
- `hide()`
- `unhide()`
- `set_focus()`
- `expand()`
- `Widget` Attribute Access

### Element Events
- Click Events
- Return Events from Inputs
- Timeout Events
- Window Close Event
- Element Visibility Change Events
- Button Bindings for Events

---

## Advanced Features

### Popup Windows
- `popup()`
- `popup_get_text()`
- `popup_get_file()`
- `popup_scrolled()`
- `popup_yes_no()`
- `popup_ok_cancel()`
- `popup_error()`
- `popup_quick_message()`
- `popup_non_blocking()`

### Themes
- Prebuilt Themes
- `theme()` to Set Theme
- Custom Theme Creation

### Persistent Data
- Save data across window closes
- `user_settings` (for settings storage)

### Window Interaction
- Multiple Windows Management
- Modal vs Non-modal Windows
- Hide and Unhide Windows

### Graphics
- Drawing in `Graph`
- Using `Canvas` Element
- Animation Techniques

### Async / Threads
- `perform_long_operation()`
- Managing GUI during Background Tasks
- Thread Safety

### Files and Folders Browsing
- File Save
- File Open
- Folder Browse

### Menus
- Menubar Element
- ButtonMenu Element
- Context Menus

### Tables and Trees
- Dynamic Table Data
- Tree Structures
- Editable Tables

---

## Programming Practices

### Error Handling
- Debugging GUIs
- Exception Capture
- Graceful Closing

### Design Patterns
- Single Event Loop
- Event Filtering
- Window Finalization

### Code Organization
- Separating Layout and Logic
- Reusing Windows
- Creating Custom Elements

### Cross-Platform Notes
- Windows, macOS, Linux Considerations
- Platform-Specific Tweaks

---

## Specialized Concepts

### Element Metadata
- Using Metadata to Attach Extra Info to Elements

### Update vs Recreate Windows
- Performance Considerations

### Scrollable Layouts
- Scrollable Frames
- Scrollable Columns

### Resizable Windows
- Fixed vs Resizable Windows
- Expand Elements to Fill Window

### Internationalization
- Unicode Support
- Font Choices for Language Compatibility

---
