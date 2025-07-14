## **State Management in Streamlit**

Streamlit apps rerun from top to bottom on every interaction. **State management** ensures values (like inputs, counters, etc.) persist across reruns, enabling advanced interactions like multi-step forms, toggles, counters, etc.

---

## `st.session_state`

A special dictionary-like object to store values across reruns.

### **Use Cases**
- Store user inputs
- Preserve computed values
- Maintain counters
- Control UI components visibility

---

## **Creating & Accessing State Variables**

```python
# Initialize variable only once
if "count" not in st.session_state:
    st.session_state.count = 0

# Access or modify
st.write("Count:", st.session_state.count)
```

---

## **Updating State Variables**

### **Using a Button**
```python
if st.button("Increment"):
    st.session_state.count += 1
```

### **Using Input Widgets**
Each widget can directly update a session variable:

```python
st.text_input("Name", key="user_name")
st.write("Hello", st.session_state.user_name)
```

---

## **Widget Keys and State**

Widgets auto-bind to `st.session_state` when given a `key`.

```python
st.checkbox("Accept", key="accepted")
if st.session_state.accepted:
    st.success("Thanks!")
```

---

## **Callback Functions**

Use `on_click`, `on_change` to run functions when widget state changes.

```python
def reset_count():
    st.session_state.count = 0

st.button("Reset", on_click=reset_count)
```

Or with widgets:

```python
def name_changed():
    st.write("New name:", st.session_state.name)

st.text_input("Enter name", key="name", on_change=name_changed)
```

---

## **Session State Operations**

| Operation                         | Example                                 |
|----------------------------------|-----------------------------------------|
| Read value                       | `st.session_state["name"]`              |
| Set value                        | `st.session_state["name"] = "Alice"`    |
| Check if key exists              | `"name" in st.session_state`            |
| Delete value                     | `del st.session_state["name"]`          |
| List all keys                    | `list(st.session_state.keys())`         |
| Clear all values                 | `st.session_state.clear()`              |

---

## **Conditional UI with State**

```python
if "show_text" not in st.session_state:
    st.session_state.show_text = False

if st.button("Toggle"):
    st.session_state.show_text = not st.session_state.show_text

if st.session_state.show_text:
    st.write("You toggled me on!")
```

---

## **Storing Complex Objects**

You can store any Python object in `st.session_state`.

```python
if "dataframe" not in st.session_state:
    st.session_state.dataframe = pd.DataFrame(np.random.randn(5, 3))

st.dataframe(st.session_state.dataframe)
```

---

## **Best Practices**

| Tip                                       | Description                                 |
|------------------------------------------|---------------------------------------------|
| Use `key` for every widget                | Ensures binding with `session_state`        |
| Always check with `if key not in state`  | Prevents overwrite during reruns            |
| Use callbacks for separation of logic    | Cleaner updates and modular code            |
| Avoid heavy operations inside callbacks  | Keeps UI responsive                         |

---
