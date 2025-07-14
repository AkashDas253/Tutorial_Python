## Element Methods in PySimpleGUI

Element methods allow you to interact programmatically with GUI components in PySimpleGUI. These methods enable actions like updating values, retrieving information, and modifying element states during runtime.

---

### Common Element Methods

1. **`update()`**:
   - **Description**: Updates the properties or values of an element. It can be used to modify the element's text, font, background color, visibility, and more.
   - **Usage**: Call this method when you need to dynamically update an element's state after the window has been created.
   - **Parameters**:
     - `text`: Set new text for the element (e.g., button label, text).
     - `visible`: Set whether the element is visible (`True` or `False`).
     - `disabled`: Set whether the element is disabled (`True` or `False`).
     - `value`: Set the new value for input elements like `InputText`, `Checkbox`, etc.
     - `size`: Change the size (width, height) of an element.
     - `font`: Set a new font for the element.
   - **Example**:
     ```python
     sg.Button('Submit', key='-SUBMIT-')  # Button creation
     window['-SUBMIT-'].update(text='Confirmed', disabled=True)
     ```

2. **`get()`**:
   - **Description**: Retrieves the current value of an element (commonly used for input fields, checkboxes, etc.).
   - **Usage**: Used to get the current data or state of an element, typically after user interaction.
   - **Example**:
     ```python
     input_value = window['-INPUT-'].get()  # Get value from input field
     ```

3. **`Widget`** (accessing the underlying tkinter widget):
   - **Description**: Provides access to the underlying tkinter widget for advanced functionality that may not be directly supported by PySimpleGUI.
   - **Usage**: Use this method when you need to perform operations that aren't exposed by PySimpleGUI but are possible with the base tkinter widget.
   - **Example**:
     ```python
     widget = window['-INPUT-'].Widget  # Access tkinter widget
     widget.config(bg='yellow')  # Modify the background color of the widget
     ```

4. **`bind()`**:
   - **Description**: Binds a function or event to an element, enabling more complex interaction or triggering additional functionality.
   - **Usage**: Bind functions to elements like buttons or inputs to perform custom actions based on events.
   - **Example**:
     ```python
     def on_button_click(event, values):
         print("Button clicked")
     window['-BUTTON-'].bind('<Button-1>', on_button_click)
     ```

5. **`set_focus()`**:
   - **Description**: Sets the focus on the specified element, meaning it becomes the active element for user input.
   - **Usage**: Used to guide user interaction by focusing on an element when required.
   - **Example**:
     ```python
     window['-INPUT-'].set_focus()  # Set focus to the input element
     ```

6. **`hide()`**:
   - **Description**: Hides an element from the window. It doesn't remove the element but makes it invisible.
   - **Usage**: Used to dynamically hide elements without destroying them.
   - **Example**:
     ```python
     window['-BUTTON-'].hide()  # Hide the button
     ```

7. **`unhide()`**:
   - **Description**: Makes a previously hidden element visible again.
   - **Usage**: Used to show elements that were previously hidden with the `hide()` method.
   - **Example**:
     ```python
     window['-BUTTON-'].unhide()  # Show the button again
     ```

8. **`select()`**:
   - **Description**: Used to select the element’s value (applicable to elements like `Checkbox`, `Radio`, `Listbox`).
   - **Usage**: Select a checkbox, radio button, or list item programmatically.
   - **Example**:
     ```python
     window['-CHECKBOX-'].select()  # Select the checkbox
     ```

9. **`deselect()`**:
   - **Description**: Deselects an element, such as a checkbox or radio button.
   - **Usage**: Deselect elements programmatically, e.g., uncheck a checkbox.
   - **Example**:
     ```python
     window['-CHECKBOX-'].deselect()  # Deselect the checkbox
     ```

10. **`toggle()`**:
    - **Description**: Toggles the state of an element. For checkboxes and radio buttons, it switches between selected and deselected.
    - **Usage**: Convenient for elements like checkboxes when you want to switch their state.
    - **Example**:
      ```python
      window['-CHECKBOX-'].toggle()  # Toggle checkbox state
      ```

11. **`expand()`**:
    - **Description**: Expands the size of an element within a layout. It helps to adjust the element size dynamically.
    - **Usage**: Used when you want to expand the space taken by an element.
    - **Example**:
      ```python
      window['-TEXT-'].expand()  # Expand the Text element
      ```

12. **`disable()`**:
    - **Description**: Disables an element, preventing it from receiving user interaction.
    - **Usage**: Use this to disable interactive elements like buttons and inputs.
    - **Example**:
      ```python
      window['-BUTTON-'].disable()  # Disable the button
      ```

13. **`enable()`**:
    - **Description**: Enables an element that was previously disabled.
    - **Usage**: Used to re-enable a previously disabled element.
    - **Example**:
      ```python
      window['-BUTTON-'].enable()  # Enable the button
      ```

14. **`click()`**:
    - **Description**: Simulates a click event on the element. It’s often used to programmatically trigger button clicks.
    - **Usage**: Trigger button clicks or other clickable elements without user input.
    - **Example**:
      ```python
      window['-BUTTON-'].click()  # Simulate a button click
      ```

15. **`bind_event()`**:
    - **Description**: Allows binding a custom event handler to a specific element or interaction.
    - **Usage**: Use this method to handle custom events that PySimpleGUI does not natively support.
    - **Example**:
      ```python
      def custom_event_handler(event, values):
          print("Custom event triggered")
      window['-BUTTON-'].bind_event('<Button-3>', custom_event_handler)
      ```

---

### Summary

Element methods in PySimpleGUI provide the ability to modify and interact with GUI elements at runtime. These methods help control the state, appearance, and behavior of elements, making it possible to create dynamic and interactive applications. Common methods include `update()`, `get()`, `hide()`, `unhide()`, `set_focus()`, and `disable()`, among others. These methods allow you to programmatically control elements to match your application's logic and user interaction needs.