## **Input Widgets in Streamlit**

Input widgets in Streamlit enable users to interact with the application by providing data or making selections. Below is a comprehensive list of input widgets along with their parameters and examples.

---

### **1. Button**

| **Description** | Creates a button. When clicked, executes the specified function or triggers an event. |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.button()`                                                                       |
| **Parameters**   |                                                                                     |
| `label`          | (str) Text on the button.                                                          |
| `key`            | (str, optional) Unique key for the widget.                                         |
| `on_click`       | (callable, optional) Callback function to execute on click.                        |
| `args`           | (tuple, optional) Arguments for `on_click` function.                              |
| **Example**      |                                                                                     |
```python
if st.button("Click Me"):
    st.write("Button clicked!")
```

---

### **2. Checkbox**

| **Description** | Creates a checkbox. Used for binary (True/False) input.                              |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.checkbox()`                                                                     |
| **Parameters**   |                                                                                     |
| `label`          | (str) Text next to the checkbox.                                                   |
| `value`          | (bool, optional) Initial state (default: False).                                   |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
agree = st.checkbox("I agree")
if agree:
    st.write("Thank you for agreeing!")
```

---

### **3. Radio Button**

| **Description** | Creates a group of radio buttons for single selection.                              |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.radio()`                                                                        |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the radio button group.                                            |
| `options`        | (list) List of options to choose from.                                             |
| `index`          | (int, optional) Default selected option index.                                     |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
choice = st.radio("Choose an option", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {choice}")
```

---

### **4. Selectbox**

| **Description** | Creates a dropdown menu for single selection.                                       |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.selectbox()`                                                                    |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the dropdown.                                                     |
| `options`        | (list) List of options to choose from.                                             |
| `index`          | (int, optional) Default selected option index.                                     |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
option = st.selectbox("Choose an option", ["Option A", "Option B", "Option C"])
st.write(f"You selected: {option}")
```

---

### **5. Multiselect**

| **Description** | Creates a dropdown menu for multiple selections.                                    |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.multiselect()`                                                                 |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the dropdown.                                                     |
| `options`        | (list) List of options to choose from.                                             |
| `default`        | (list, optional) Pre-selected options.                                             |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
selected_options = st.multiselect("Choose options", ["Option 1", "Option 2", "Option 3"])
st.write(f"You selected: {selected_options}")
```

---

### **6. Slider**

| **Description** | Creates a slider for numeric or date input.                                         |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.slider()`                                                                      |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the slider.                                                       |
| `min_value`      | (int/float/datetime) Minimum value.                                                |
| `max_value`      | (int/float/datetime) Maximum value.                                                |
| `value`          | (int/float/datetime, optional) Initial value.                                      |
| `step`           | (int/float, optional) Step size.                                                  |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
age = st.slider("Select your age", 0, 100, 25)
st.write(f"Your age is: {age}")
```

---

### **7. Text Input**

| **Description** | Creates a single-line text input box.                                               |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.text_input()`                                                                  |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the input box.                                                    |
| `value`          | (str, optional) Default value.                                                    |
| `max_chars`      | (int, optional) Maximum number of characters allowed.                             |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
name = st.text_input("Enter your name")
st.write(f"Hello, {name}!")
```

---

### **8. Text Area**

| **Description** | Creates a multi-line text input box.                                               |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.text_area()`                                                                   |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the text area.                                                    |
| `value`          | (str, optional) Default text.                                                     |
| `height`         | (int, optional) Height of the text area (in pixels).                              |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
feedback = st.text_area("Provide your feedback")
st.write(f"Your feedback: {feedback}")
```

---

### **9. Number Input**

| **Description** | Creates a numeric input box for integers or floats.                                 |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.number_input()`                                                                |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the input box.                                                    |
| `min_value`      | (int/float, optional) Minimum value.                                               |
| `max_value`      | (int/float, optional) Maximum value.                                               |
| `value`          | (int/float, optional) Default value.                                               |
| `step`           | (int/float, optional) Increment size.                                              |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
num = st.number_input("Enter a number", min_value=0, max_value=100, value=10, step=1)
st.write(f"You entered: {num}")
```

---

### **10. File Uploader**

| **Description** | Allows users to upload files.                                                      |
|------------------|--------------------------------------------------------------------------------------|
| **Function**     | `st.file_uploader()`                                                               |
| **Parameters**   |                                                                                     |
| `label`          | (str) Label for the file uploader.                                                |
| `type`           | (str/list, optional) Allowed file types (e.g., "csv", "txt").                      |
| `accept_multiple_files` | (bool, optional) Allow multiple file uploads.                               |
| `key`            | (str, optional) Unique key for the widget.                                         |
| **Example**      |                                                                                     |
```python
file = st.file_uploader("Upload a file", type=["csv", "txt"])
if file:
    st.write(f"Uploaded file name: {file.name}")
```

---
