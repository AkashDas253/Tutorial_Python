## Element Attributes in PySimpleGUI

Element attributes define the properties and behavior of GUI components (elements) such as buttons, text, sliders, etc. These attributes help customize the appearance and functionality of each element in a PySimpleGUI window. Below is a breakdown of common element attributes in PySimpleGUI:

---

### Common Element Attributes

1. **Key**:
   - **Description**: The `key` is a unique identifier for the element. It allows access to the element's value and can be used to update or retrieve its state during execution.
   - **Usage**: 
     - Important for dynamic updates.
     - Can be used in events to trigger actions based on element interactions.
   - **Example**:
     ```python
     sg.Button('Submit', key='-SUBMIT-')
     ```

2. **Size**:
   - **Description**: Specifies the dimensions (width and height) of an element. The `size` attribute takes a tuple of `(width, height)`, which controls the size of the element.
   - **Usage**: Adjust the size of buttons, input fields, etc.
   - **Example**:
     ```python
     sg.Button('Click Me', size=(20, 2))  # Width of 20, height of 2
     ```

3. **Font**:
   - **Description**: Sets the font of the text displayed within the element. It takes a string in the format `"font_name font_size"`.
   - **Usage**: Customize the text appearance.
   - **Example**:
     ```python
     sg.Text('Hello, world!', font='Arial 14')
     ```

4. **Text Color**:
   - **Description**: Defines the color of the text in an element. This can be set using a color name or a hex code.
   - **Usage**: Customize text color for better visibility or aesthetics.
   - **Example**:
     ```python
     sg.Text('Important Message', text_color='red')
     ```

5. **Background Color**:
   - **Description**: Defines the background color of an element. This can be set using a color name or hex code.
   - **Usage**: Change the background color for better UI design.
   - **Example**:
     ```python
     sg.Button('OK', button_color=('white', 'blue'))
     ```

6. **Pad**:
   - **Description**: Specifies the padding around the element. The `pad` attribute takes a tuple `(vertical_padding, horizontal_padding)` to control the spacing around the element.
   - **Usage**: Space out elements for better UI layout.
   - **Example**:
     ```python
     sg.Button('Click Me', pad=(10, 10))
     ```

7. **Visible**:
   - **Description**: Controls whether an element is visible on the window. Set this attribute to `False` to hide an element and `True` to show it.
   - **Usage**: Dynamically hide or show elements based on conditions.
   - **Example**:
     ```python
     sg.Button('Hidden Button', visible=False)
     ```

8. **Disabled**:
   - **Description**: When set to `True`, this attribute disables the element, making it unresponsive to user interaction. The default value is `False`.
   - **Usage**: Use for buttons, inputs, and other elements that need to be temporarily disabled.
   - **Example**:
     ```python
     sg.Button('Disabled Button', disabled=True)
     ```

9. **Tooltip**:
   - **Description**: Adds a tooltip to an element. The tooltip is displayed when the user hovers over the element with the mouse.
   - **Usage**: Provide additional information about an element.
   - **Example**:
     ```python
     sg.Button('Hover Me', tooltip='Click this button to submit')
     ```

10. **Default Button**:
    - **Description**: Defines a button as the default one that is activated when the user presses the Enter key.
    - **Usage**: Useful for making a primary action more accessible.
    - **Example**:
      ```python
      sg.Button('Submit', default_button=True)
      ```

11. **Image Size**:
    - **Description**: Specifies the size of the image displayed in an `Image` element.
    - **Usage**: Control the dimensions of an image.
    - **Example**:
      ```python
      sg.Image(filename='image.png', size=(200, 200))
      ```

12. **Justification**:
    - **Description**: Defines how text within an element is aligned. It can be set to `'left'`, `'center'`, or `'right'`.
    - **Usage**: Adjust text alignment inside elements like `Text`, `Button`, and `InputText`.
    - **Example**:
      ```python
      sg.Text('Centered Text', justification='center')
      ```

13. **Border Width**:
    - **Description**: Specifies the width of the border around an element. This applies to elements like `Button`, `InputText`, and `Text`.
    - **Usage**: Customize the visual appearance by adjusting the border thickness.
    - **Example**:
      ```python
      sg.Button('Click Me', border_width=2)
      ```

14. **Enabled**:
    - **Description**: Similar to `Disabled`, but it explicitly defines whether an element is enabled or not. When set to `False`, the element becomes unresponsive to user actions.
    - **Usage**: For controlling the active state of an element.
    - **Example**:
      ```python
      sg.Button('Click Me', enabled=False)
      ```

15. **Auto Size**:
    - **Description**: If set to `True`, this attribute automatically adjusts the size of an element based on its content.
    - **Usage**: Ensure elements resize dynamically to fit their content.
    - **Example**:
      ```python
      sg.Button('Click Me', auto_size_button=True)
      ```

16. **Change Submits**:
    - **Description**: For `InputText` and `Multiline` elements, this attribute controls whether the element submits data on every change.
    - **Usage**: Use it when you want to capture the value in real-time as the user types.
    - **Example**:
      ```python
      sg.InputText(change_submits=True)
      ```

17. **Text Color**:
    - **Description**: Sets the color of the text displayed in the element.
    - **Usage**: Control the text color for visibility or aesthetic purposes.
    - **Example**:
      ```python
      sg.Text('Red Text', text_color='red')
      ```

---

### Specialized Element Attributes

1. **Button Color**:
   - **Description**: Defines the color of the button text and its background. It takes a tuple of two color values: one for the text and one for the background.
   - **Usage**: Customize the color scheme of buttons.
   - **Example**:
     ```python
     sg.Button('Click Me', button_color=('black', 'yellow'))
     ```

2. **Values**:
   - **Description**: Holds the current values of input elements (e.g., `InputText`, `Combo`, `Listbox`, `Checkbox`).
   - **Usage**: This attribute is used when interacting with the user’s input and can be updated or retrieved dynamically.
   - **Example**:
     ```python
     sg.InputText(default_text='Enter something', key='-INPUT-')
     values = window['-INPUT-'].get()  # Retrieve value
     ```

3. **Font Size**:
   - **Description**: Controls the size of the font used in an element, typically applied to text-heavy elements like `Text`, `Button`, and `InputText`.
   - **Usage**: Adjust text size to suit the design or readability needs.
   - **Example**:
     ```python
     sg.Text('Large Text', font=('Helvetica', 20))
     ```

4. **Relief**:
   - **Description**: Defines the visual effect or border style of an element. It can be set to `'flat'`, `'raised'`, `'sunken'`, `'solid'`, or `'groove'`.
   - **Usage**: Add visual depth or borders around elements.
   - **Example**:
     ```python
     sg.Button('Click Me', relief='raised')
     ```

---

### Summary

Element attributes in PySimpleGUI allow you to control the appearance, behavior, and interaction of each GUI component. Key attributes include `key`, `size`, `font`, `text_color`, `background_color`, `pad`, and more. These attributes allow for flexibility in designing and customizing the layout and interaction within the window.