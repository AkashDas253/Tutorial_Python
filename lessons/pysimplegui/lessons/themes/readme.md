## Themes in PySimpleGUI

Themes in PySimpleGUI allow you to easily apply predefined styles to your GUI elements, making it easier to create visually appealing applications without manually adjusting individual element properties. Themes control the overall color scheme and style of the application, ensuring consistency and saving development time.

---

### Key Concepts

1. **Theme Application**:
   - A theme is a predefined set of colors and font styles that are applied to all elements in a window.
   - The theme affects the color of the text, buttons, input fields, and other graphical components.
   - Themes can be applied globally for all windows or to a specific window.

2. **Theme Function**:
   - **Function**: `sg.theme('ThemeName')`
   - The theme is applied when the program starts, but it can also be changed dynamically during runtime.
   - **Example**:
     ```python
     sg.theme('DarkGrey5')  # Apply a theme
     ```

3. **Available Themes**:
   PySimpleGUI comes with several built-in themes that you can choose from. Some common ones are:
   - 'DarkGrey5'
   - 'LightGreen'
   - 'DarkBlue3'
   - 'SystemDefault'
   - 'Topanga'
   - 'Reddit'
   - 'BrownBlue'
   - 'BlueMono'
   - 'GreenTan'
   
   You can see the full list of available themes with the command:
   ```python
   print(sg.list_of_themes())  # List all available themes
   ```

---

### Theme Elements

Each theme defines the following visual properties:

- **Background Color**: Controls the background color of the window and elements.
- **Text Color**: Sets the color for text elements like `sg.Text()`.
- **Button Color**: Controls the background and text color of buttons (`sg.Button()`).
- **Input Field Color**: Specifies the background and text color for input fields (`sg.InputText()`).
- **Other Elements**: Other graphical elements like checkboxes, sliders, and progress bars also follow the theme's visual style.

---

### Changing Themes Dynamically

You can change the theme dynamically during runtime. This allows you to create interactive applications where users can select or switch between different themes.

- **Function**: `sg.theme()`
- **Example**:
  ```python
  sg.theme('DarkBlue3')  # Set initial theme
  layout = [
      [sg.Text('This is a themed window')],
      [sg.Button('OK')]
  ]
  window = sg.Window('Themed Window', layout)
  
  event, values = window.read()
  if event == sg.WIN_CLOSED:
      window.close()
  
  sg.theme('LightGreen')  # Change theme dynamically
  window.close()
  ```

---

### Custom Themes

You can define your own custom themes by specifying the colors and styles for various elements.

- **Function**: `sg.LOOK_AND_FEEL_TABLE`
- **Steps**:
  1. Define a dictionary with the custom theme settings.
  2. Use `sg.theme_add_new()` to register the theme.
  
- **Example**:
  ```python
  custom_theme = {
      'BACKGROUND': '#282828',
      'TEXT': '#FFFFFF',
      'INPUT': '#2E2E2E',
      'BUTTON': ('#FFFFFF', '#4CAF50'),
      'BUTTON_HIGHLIGHT': '#45A049',
      'TEXT_INPUT': ('#FFFFFF', '#333333'),
      'SCROLL': '#3E3E3E',
      'LABEL': '#FFFFFF',
      'PROGRESS': '#0078D7',
  }
  
  sg.theme_add_new('CustomTheme', custom_theme)
  sg.theme('CustomTheme')

  layout = [
      [sg.Text('This is a custom themed window')],
      [sg.Button('OK')]
  ]
  window = sg.Window('Custom Theme Window', layout)
  window.read()
  window.close()
  ```

In the example above, we created a custom theme by specifying color values for various UI elements. The colors defined are applied to the entire window or specific elements like buttons or text fields.

---

### Theme-Specific Customization

While themes apply a global style to all elements, you can still modify individual element attributes if needed. For example, you can change the background color of a button, even when a theme is applied.

- **Example**:
  ```python
  sg.theme('DarkGrey5')  # Apply a theme
  layout = [
      [sg.Button('Custom Button', button_color=('white', 'red'))]  # Override button color
  ]
  window = sg.Window('Custom Button', layout)
  window.read()
  window.close()
  ```

---

### Summary

Themes in PySimpleGUI provide a quick and easy way to apply a consistent visual style to your GUI. They come with several predefined options, and you can create custom themes by specifying your preferred colors and styles. Themes can be applied globally or changed dynamically during runtime. Additionally, individual element attributes can still be customized, even when a theme is in use, giving you flexibility in your design.

