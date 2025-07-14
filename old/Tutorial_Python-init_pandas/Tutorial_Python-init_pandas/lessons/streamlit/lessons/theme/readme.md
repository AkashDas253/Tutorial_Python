## **Themes in Streamlit**

#### **Setting Themes in Streamlit**

- Streamlit allows users to set a custom theme for the entire app. 
- The theme can be configured using: 
    - `st.set_page_config()` or 
    - a `.streamlit/config.toml` file.
- You can change properties like the primary color, background color, font styles, and more.

##### **1. Using `st.set_page_config()`**

| **Function**                   | **Description**                                                                                      | **Parameters**                                                                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `st.set_page_config()`          | Set the page's theme and layout properties.                                                           | `page_title (str)`<br> Default: None<br> `page_icon (str or PIL.Image)`<br> Default: None<br> `layout (str)`<br> Default: "centered"<br> `initial_sidebar_state (str)`<br> Default: "auto" |

Example:
```python
st.set_page_config(page_title="My Streamlit App", layout="wide", initial_sidebar_state="expanded")
```

#### **2. Customizing Themes via `config.toml`**

You can also customize the theme in Streamlit by editing the `.streamlit/config.toml` file. 

| **Property**                    | **Description**                                                                                      | **Possible Values**                                                                                                                 |
|----------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `theme.primaryColor`             | Primary color for the app's user interface.                                                          | Any valid CSS color (e.g., `"#FF6347"`, `"blue"`, `"rgb(255, 99, 71)"`)                                                            |
| `theme.backgroundColor`          | Background color for the app.                                                                        | Any valid CSS color (e.g., `"#F4F4F4"`, `"white"`, `"rgb(244, 244, 244)"`)                                                        |
| `theme.secondaryBackgroundColor` | Background color for elements like sidebars and widgets.                                             | Any valid CSS color (e.g., `"#FFFFFF"`, `"gray"`, `"rgb(240, 240, 240)"`)                                                          |
| `theme.textColor`                | Text color for the app.                                                                               | Any valid CSS color (e.g., `"#000000"`, `"black"`, `"rgb(0, 0, 0)"`)                                                              |
| `theme.font`                     | Font style for the app's text.                                                                       | `"sans serif"`, `"serif"`, `"monospace"`, or custom Google fonts (e.g., `"Roboto"`)                                                 |
| `theme.textFont`                 | Font style for text-based widgets like `st.text_input()`.                                             | `"sans serif"`, `"serif"`, `"monospace"`, or custom Google fonts (e.g., `"Roboto"`)                                                 |
| `theme.headingFont`              | Font style for headings.                                                                              | `"sans serif"`, `"serif"`, `"monospace"`, or custom Google fonts (e.g., `"Roboto"`)                                                 |
| `theme.codeFont`                 | Font style for code elements in the app (e.g., `st.code()`).                                          | `"monospace"` or custom fonts (e.g., `"Courier New"`)                                                                               |

##### **Example: `config.toml`**

```toml
[theme]
primaryColor = "#FF6347"
backgroundColor = "#F4F4F4"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#000000"
font = "sans serif"
textFont = "Arial"
headingFont = "Courier New"
codeFont = "Courier"
```

#### **3. Example Custom Themes in Streamlit**

You can experiment with different combinations to match the branding or aesthetics of your app. For example:

| **Theme Property**              | **Example Value**        | **Result**                                                                                                   |
|----------------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------|
| `theme.primaryColor`             | `"#FF6347"`              | App's primary color will be tomato red.                                                                       |
| `theme.backgroundColor`          | `"#F4F4F4"`              | Light gray background color for the app.                                                                     |
| `theme.secondaryBackgroundColor` | `"#FFFFFF"`              | White background for sidebars and widgets.                                                                   |
| `theme.textColor`                | `"#000000"`              | Black text color.                                                                                           |
| `theme.font`                     | `"sans serif"`           | General text font as sans-serif (default system font).                                                       |
| `theme.textFont`                 | `"Arial"`                | Font for text input widgets set to Arial.                                                                   |
| `theme.headingFont`              | `"Courier New"`          | Headings will be displayed in the Courier New font.                                                          |
| `theme.codeFont`                 | `"Courier"`              | Code elements will use the Courier font, commonly used for displaying monospaced text.                       |

---
