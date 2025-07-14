## **Components**

---

#### **Text Display**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.title()`                    | Displays a large title at the top of the page.                                                       | `title (str)`<br> Default: None                                                                                                                   |
| `st.header()`                   | Displays a header text in a slightly smaller font.                                                   | `label (str)`<br> Default: None                                                                                                                   |
| `st.subheader()`                | Displays a subheader, smaller than the header.                                                       | `label (str)`<br> Default: None                                                                                                                   |
| `st.markdown()`                 | Displays Markdown-formatted text.                                                                    | `body (str)`<br> Default: None<br> `unsafe_allow_html (bool)`<br> Default: `False`                                                               |
| `st.caption()`                  | Displays text in a caption style, used for descriptions or smaller text.                             | `label (str)`<br> Default: None                                                                                                                   |
| `st.code()`                     | Displays code in a formatted style.                                                                  | `code (str)`<br> Default: None<br> `language (str)`<br> Default: 'python'                                                                          |
| `st.latex()`                    | Displays LaTeX-formatted text for mathematical expressions.                                          | `body (str)`<br> Default: None                                                                                                                   |

Example:
```python
st.title("Streamlit Title")
st.code("print('Hello, World!')")
```

---

#### **Data Display**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.write()`                    | Displays a wide variety of content, including text, dataframes, and more.                             | `data`<br> Default: None<br> `use_container_width (bool)`<br> Default: `False`                                                                     |
| `st.json()`                     | Displays JSON data in a readable format.                                                             | `json (dict or str)`<br> Default: None<br> `expanded (bool)`<br> Default: `True`                                                                   |
| `st.dataframe()`                | Displays a Pandas DataFrame in a scrollable table.                                                   | `data (pandas.DataFrame)`<br> Default: None<br> `width (int)`<br> Default: None<br> `height (int)`<br> Default: None                               |
| `st.table()`                    | Displays a table of data, similar to `st.dataframe()` but without interactive features.              | `data (pd.DataFrame or list)`<br> Default: None                                                                                                    |
| `st.metric()`                   | Displays key metrics with a label, value, and (optionally) a delta (change).                         | `label (str)`<br> Default: None<br> `value (str or int or float)`<br> Default: None<br> `delta (str or int or float)`<br> Default: None             |

Example:
```python
st.write("Here is some text")
st.dataframe(df)
```

---

#### **Media**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.image()`                    | Displays an image. Supports multiple formats like PNG, JPG, etc.                                      | `image (str or bytes)`<br> Default: None<br> `caption (str)`<br> Default: None<br> `use_column_width (bool)`<br> Default: `False`                   |
| `st.audio()`                    | Displays an audio player, allowing the user to listen to audio files.                                | `audio (str or bytes)`<br> Default: None<br> `format (str)`<br> Default: 'audio/wav'                                                               |
| `st.video()`                    | Displays a video player to show video files.                                                         | `video (str or bytes)`<br> Default: None<br> `format (str)`<br> Default: 'video/mp4'                                                               |

Example:
```python
st.image("image.png")
st.audio("audio.mp3")
```

---

#### **User Input Widgets**

##### **Text Input**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.text_input()`               | Displays a text box for user input.                                                                   | `label (str)`<br> Default: None<br> `value (str)`<br> Default: ""<br> `max_chars (int)`<br> Default: None                                           |
| `st.text_area()`                | Displays a larger text area for longer inputs.                                                       | `label (str)`<br> Default: None<br> `value (str)`<br> Default: ""<br> `height (int)`<br> Default: 200                                              |

##### **Numeric Input**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.number_input()`             | Displays a numeric input field, where the user can specify a value.                                  | `label (str)`<br> Default: None<br> `min_value (int or float)`<br> Default: None<br> `max_value (int or float)`<br> Default: None<br> `step`<br> Default: 1 |

##### **Date/Time Input**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.date_input()`               | Displays a date picker for users to select a date.                                                   | `label (str)`<br> Default: None<br> `value (datetime.date)`<br> Default: current date                                                              |
| `st.time_input()`               | Displays a time picker for users to select a time.                                                   | `label (str)`<br> Default: None<br> `value (datetime.time)`<br> Default: current time                                                              |

##### **Selections**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.selectbox()`                | Displays a drop-down menu for a single selection.                                                     | `label (str)`<br> Default: None<br> `options (list)`<br> Default: []<br> `index (int)`<br> Default: 0                                                |
| `st.multiselect()`              | Displays a list of options, allowing the user to select multiple.                                     | `label (str)`<br> Default: None<br> `options (list)`<br> Default: []<br> `default (list)`<br> Default: []                                           |
| `st.radio()`                    | Displays a set of radio buttons for single selection.                                                 | `label (str)`<br> Default: None<br> `options (list)`<br> Default: []<br> `index (int)`<br> Default: 0                                              |
| `st.checkbox()`                 | Displays a checkbox for binary choices.                                                               | `label (str)`<br> Default: None<br> `value (bool)`<br> Default: False                                                                              |

##### **Sliders**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.slider()`                   | Displays a slider for selecting a value in a range.                                                   | `label (str)`<br> Default: None<br> `min_value (int or float)`<br> Default: 0<br> `max_value (int or float)`<br> Default: 100                        |
| `st.select_slider()`            | Displays a slider with select options for non-continuous values.                                      | `label (str)`<br> Default: None<br> `options (list)`<br> Default: []<br> `index (int)`<br> Default: 0                                             |

##### **File Upload**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.file_uploader()`            | Allows users to upload files into the app.                                                           | `label (str)`<br> Default: None<br> `type (list)`<br> Default: None<br> `accept_multiple_files (bool)`<br> Default: False                             |

##### **Buttons**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.button()`                   | Displays a button that can trigger actions when clicked.                                             | `label (str)`<br> Default: None<br> `key (str)`<br> Default: None                                                                                 |
| `st.form_submit_button()`       | Displays a submit button inside a form, used to handle grouped input.                                | `label (str)`<br> Default: None<br> `use_container_width (bool)`<br> Default: `False`                                                             |

##### **Others**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.color_picker()`             | Displays a color picker for users to select a color.                                                 | `label (str)`<br> Default: None<br> `value (str)`<br> Default: "#ffffff"                                                                            |

---
