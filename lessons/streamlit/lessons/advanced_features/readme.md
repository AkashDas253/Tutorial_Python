## Advanced Features in Streamlit

Streamlit supports several advanced capabilities beyond basic UI and interaction. These features enable building powerful, customizable, and modular apps.

---

## Custom Components

Streamlit allows integration of custom frontend elements via JavaScript/TypeScript using the Component API.

| Feature              | Description                                       |
|----------------------|---------------------------------------------------|
| `streamlit.components.v1` | Create and render custom components         |
| `declare_component()`     | Define the frontend module                   |
| `components.html()`       | Embed raw HTML/JS                           |

**Example:**
```python
import streamlit.components.v1 as components
components.html("<button>Click Me</button>", height=50)
```

---

## Multipage Apps

Multiple pages can be created in a project using separate Python files inside a `pages/` directory.

| Feature                  | Description                                       |
|--------------------------|---------------------------------------------------|
| `st.sidebar.radio()`     | Used for manual navigation (older way)           |
| File structure-based nav | Streamlit detects files inside `/pages/` folder  |

**Structure:**
```
my_app/
│
├── Home.py
├── pages/
│   ├── Page1.py
│   └── Page2.py
```

---

## Forms

Streamlit supports grouping inputs inside `st.form()` to submit multiple values at once.

| Feature        | Description                                     |
|----------------|-------------------------------------------------|
| `st.form()`    | Wraps inputs                                    |
| `st.form_submit_button()` | Triggers form submission           |

**Example:**
```python
with st.form("login"):
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login")

if submit:
    st.success("Submitted")
```

---

## Asynchronous Updates

Currently limited. For responsiveness:
- Use `st.spinner()` to show loading indicators
- Use `time.sleep()` to simulate waiting
- Use third-party integration (e.g., threading or async calls) cautiously

---

## Bi-directional Communication (Custom Components)

Can pass data from Python to JS and back using custom components.

| Direction     | Tool                              |
|----------------|----------------------------------|
| Python → JS   | Component arguments               |
| JS → Python   | Return values via `declare_component` |

---

## File Upload and Processing

Use `st.file_uploader()` to accept files (images, CSV, etc.) and process using pandas, PIL, etc.

| Parameter     | Purpose                         |
|---------------|----------------------------------|
| `type`        | Restrict file extensions         |
| `accept_multiple_files` | Enable batch uploads |

---

## Caching and Performance

| Method               | Description                                          |
|----------------------|------------------------------------------------------|
| `@st.cache_data`     | Cache pure functions that return data (default TTL) |
| `@st.cache_resource` | Cache models or sessions (e.g., DB, ML models)      |

**Example:**
```python
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")
```

---

## Theming via Config

Theme can be set via `config.toml` or programmatically with `st.set_page_config()`. Useful for building white-labeled apps.

---

## URL Parameters

Use `st.experimental_get_query_params()` and `st.experimental_set_query_params()` to read/write to URL for navigation, bookmarking, and sharing.

---

## Session State Management

Use `st.session_state` to maintain values across reruns.

| Feature              | Description                         |
|----------------------|-------------------------------------|
| Value persistence    | Store flags, counters, temporary data |
| Control reruns       | Avoid repeated execution on changes |

---

## Running in Headless/Script Mode

Streamlit apps can be scheduled via CLI or embedded in larger scripts using the `streamlit run` command with `--server.headless true`.

---
