## Popup Windows in PySimpleGUI

Popup windows in PySimpleGUI provide a simple way to display messages, alerts, or get user input without needing to create a full-fledged window. They are commonly used for displaying error messages, warnings, or requesting basic user input. Popups are modal, meaning they block interaction with the main window until the user closes or responds to them.

---

### Types of Popups

1. **Text-based Popups (Message Boxes)**:
   - **Description**: Used to display a message to the user.
   - **Function**: `sg.popup()`
   - **Example**:
     ```python
     sg.popup('Hello, this is a message popup!')
     ```

2. **OK/Cancel Popups**:
   - **Description**: Popups that give the user the option to click "OK" or "Cancel."
   - **Function**: `sg.popup_ok()`, `sg.popup_cancel()`
   - **Example**:
     ```python
     sg.popup_ok('This is an OK popup!')
     sg.popup_cancel('Would you like to cancel?')
     ```

3. **Yes/No Popups**:
   - **Description**: Popups that allow the user to choose between "Yes" or "No."
   - **Function**: `sg.popup_yes_no()`
   - **Example**:
     ```python
     response = sg.popup_yes_no('Do you want to proceed?')
     if response == 'Yes':
         print('Proceeding...')
     else:
         print('Cancelled')
     ```

4. **Input Popups**:
   - **Description**: Used to prompt the user for some input, such as text or a number.
   - **Function**: `sg.popup_get_text()`, `sg.popup_get_file()`
   - **Example**:
     ```python
     user_input = sg.popup_get_text('Enter your name:')
     print(f'Hello, {user_input}!')
     ```

5. **Password Popups**:
   - **Description**: Used to prompt the user to input a password in a secure manner.
   - **Function**: `sg.popup_get_password()`
   - **Example**:
     ```python
     password = sg.popup_get_password('Enter your password:')
     print(f'Password entered: {password}')
     ```

6. **File Chooser Popups**:
   - **Description**: Used to prompt the user to choose a file from their filesystem.
   - **Function**: `sg.popup_get_file()`
   - **Example**:
     ```python
     file_path = sg.popup_get_file('Choose a file', file_types=(('Text Files', '*.txt'),))
     print(f'File selected: {file_path}')
     ```

7. **Folder Chooser Popups**:
   - **Description**: Used to prompt the user to choose a folder from their filesystem.
   - **Function**: `sg.popup_get_folder()`
   - **Example**:
     ```python
     folder_path = sg.popup_get_folder('Choose a folder')
     print(f'Folder selected: {folder_path}')
     ```

8. **Multiline Input Popups**:
   - **Description**: Allows the user to input multiple lines of text.
   - **Function**: `sg.popup_get_text()` with multiline option
   - **Example**:
     ```python
     multiline_input = sg.popup_get_text('Enter multiple lines of text:', multiline=True)
     print(f'Multiline input:\n{multiline_input}')
     ```

---

### Popup Customization

Popups in PySimpleGUI can be customized in several ways, such as defining window title, text size, background color, etc.

- **Setting Window Title**: You can set a custom title for the popup window using the `title` argument.
  - **Example**:
    ```python
    sg.popup('Hello!', title='Custom Popup Title')
    ```

- **Setting Background Color**: You can change the background color of the popup.
  - **Example**:
    ```python
    sg.popup('Hello!', background_color='lightblue')
    ```

- **Setting Button Text**: Customize button text using the `button_color` argument.
  - **Example**:
    ```python
    sg.popup_yes_no('Do you want to save your work?', button_color=('white', 'green'))
    ```

---

### Advanced Popup Usage

1. **Multiple Inputs Popup**:
   You can create a popup that accepts multiple types of input (text, checkboxes, etc.).
   - **Example**:
     ```python
     layout = [
         [sg.Text('Enter your name:'), sg.InputText(key='name')],
         [sg.Text('Enter your age:'), sg.InputText(key='age')],
         [sg.Button('OK')]
     ]
     window = sg.Window('Multiple Inputs', layout)

     event, values = window.read()
     if event == 'OK':
         print(f"Name: {values['name']}, Age: {values['age']}")
     window.close()
     ```

2. **Popup with a Progress Bar**:
   If you need a popup that includes a progress bar to show long-running tasks, you can use `sg.ProgressBar` within a custom window layout.
   - **Example**:
     ```python
     layout = [
         [sg.Text('Processing...')],
         [sg.ProgressBar(max_value=100, orientation='h', size=(20, 20), key='progress')],
     ]
     window = sg.Window('Progress', layout, finalize=True)

     for i in range(100):
         window['progress'].update(i + 1)
         window.read(timeout=10)

     window.close()
     ```

---

### Summary

Popups in PySimpleGUI provide an easy way to display messages or collect user input in a simple, non-blocking manner. They can be used for a variety of purposes, such as displaying information, gathering input, or confirming actions with users. Popups are highly customizable with different types of buttons, text formatting, and options for user input. They provide a quick and simple way to enhance the interactivity of an application without needing to build complex layouts.