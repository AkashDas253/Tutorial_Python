# Overview of **Streamlit**

Streamlit is an open-source Python framework used to build fast, interactive, and data-driven web apps with minimal effort. It's widely used in **data science**, **machine learning**, and **analytics dashboards** due to its simplicity and Python-first design.

---

## Key Characteristics

| Feature                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Language                  | Python-only (no HTML/JS needed)                                             |
| Execution Model           | Script reruns top to bottom on every interaction                           |
| Development Style         | Declarative, widget-based UI                                                |
| Deployment Options        | Streamlit Cloud, Heroku, AWS, Docker, etc.                                  |
| Target Audience           | Data scientists, ML engineers, analysts                                     |

---

## Basic Usage

Install and run:

```bash
pip install streamlit
streamlit run app.py
```

Example `app.py`:
```python
import streamlit as st
st.title("Hello Streamlit")
st.write("Welcome to your first app!")
```

---

## Core Concepts

| Concept            | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Widgets**        | UI elements like sliders, buttons, text inputs                              |
| **Layout**         | Organize using columns, containers, expanders                               |
| **Interactivity**  | Use session state and event-based widgets                                   |
| **Charts**         | Built-in support for Matplotlib, Plotly, Altair, Pydeck, etc.               |
| **Media Support**  | Display images, audio, and video                                            |
| **Caching**        | Cache results of expensive computations using `@st.cache_*` decorators      |
| **Session State**  | Store values across interactions                                            |
| **Multi-page Apps**| Organize large apps via the `/pages` directory                              |

---

## Main Components

| Type                | Functions Used                                                             |
|---------------------|----------------------------------------------------------------------------|
| **Text & Formatting** | `st.title()`, `st.header()`, `st.subheader()`, `st.markdown()`, `st.code()` |
| **Input Widgets**     | `st.button()`, `st.slider()`, `st.text_input()`, `st.radio()`               |
| **Data Display**      | `st.dataframe()`, `st.table()`, `st.json()`, `st.metric()`                  |
| **Charts**            | `st.line_chart()`, `st.bar_chart()`, `st.pyplot()`, `st.altair_chart()`    |
| **Media**             | `st.image()`, `st.audio()`, `st.video()`                                   |
| **Layout**            | `st.sidebar`, `st.columns()`, `st.expander()`, `st.container()`            |

---

## Theming & Configuration

Customize using `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
font = "monospace"
```

---

## App Structure

```plaintext
project/
│
├── app.py               # Main script
├── pages/               # Additional pages
│   └── page1.py
├── .streamlit/
│   └── config.toml      # Theming
```

---

## Deployment

| Method              | Steps                                                                 |
|---------------------|-----------------------------------------------------------------------|
| **Streamlit Cloud** | Push to GitHub → connect to Streamlit Cloud                          |
| **Heroku**          | Add `Procfile`, `requirements.txt`, deploy via Heroku CLI            |
| **Docker**          | Create `Dockerfile`, use `streamlit run` as entrypoint               |

---

## Advantages

- Python-native and beginner-friendly
- Live reload on save
- No front-end expertise needed
- Built-in support for data science libraries
- Active community and ecosystem

---

## Limitations

- Limited custom JS/CSS unless using components
- Stateless design (must use `st.session_state`)
- Not meant for full-stack apps (e.g., no database backend built-in)

---
