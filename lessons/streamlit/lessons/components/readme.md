# Types of Components in Streamlit

Streamlit provides a rich set of **built-in components** to help developers build interactive and data-driven apps. Additionally, **custom and third-party components** extend functionality.

---

## 🔹 1. **Core Built-in Components**

| Type               | Examples                                               | Description                                      |
|--------------------|--------------------------------------------------------|--------------------------------------------------|
| **Text**           | `st.text`, `st.markdown`, `st.latex`, `st.code`        | Display formatted text and code                  |
| **Data Display**   | `st.dataframe`, `st.table`, `st.json`, `st.metric`     | Show data in structured formats                  |
| **Media**          | `st.image`, `st.audio`, `st.video`                     | Render media elements                            |
| **Widgets**        | `st.button`, `st.slider`, `st.selectbox`, `st.text_input`, etc. | Input widgets for interactivity         |
| **Layout**         | `st.columns`, `st.expander`, `st.container`, `st.tabs`, `st.sidebar` | Organize layout                |
| **Charts & Viz**   | `st.line_chart`, `st.bar_chart`, `st.pyplot`, `st.plotly_chart`, etc. | Built-in and 3rd-party chart support |
| **Control Flow**   | `st.stop`, `st.form`, `st.form_submit_button`, `st.experimental_rerun` | Execution control elements        |
| **State**          | `st.session_state`                                     | Manage persistent values across reruns           |
| **Status/Feedback**| `st.progress`, `st.spinner`, `st.toast`, `st.success`, etc. | Visual feedback on operations        |

---

## 🔹 2. **Custom Components (Streamlit Components API)**

| Type                      | Description                                          |
|---------------------------|------------------------------------------------------|
| **Custom Frontend Widgets** | Built using HTML/JS (React or Vanilla) via Streamlit Components |
| **Bidirectional Communication** | Use `streamlit.components.v1` to pass data between Python and JS |
| **Embedding External Libraries** | Add sliders, maps, editors, etc., from external UI libraries |

---

## 🔹 3. **Third-party Components**

| Component                | Description                                           |
|---------------------------|-------------------------------------------------------|
| `streamlit-aggrid`        | Advanced interactive data grid                        |
| `streamlit-folium`        | Embeds Folium maps                                    |
| `streamlit-echarts`       | Use Apache ECharts visualizations                     |
| `streamlit-drawable-canvas` | Canvas for drawing/sketching with mouse input         |
| `streamlit-webrtc`        | WebRTC-based audio/video communication                |
| `streamlit-toggle-switch` | Fancy toggle UI widgets                               |

---

## 🧩 Summary by Purpose

| Purpose              | Components Used                                                |
|----------------------|----------------------------------------------------------------|
| Layout & Styling     | `st.columns`, `st.expander`, `st.sidebar`, `st.container`      |
| Data Interaction     | `st.dataframe`, `st.table`, `st.selectbox`, `st.multiselect`   |
| Visualization        | `st.line_chart`, `st.map`, `st.pyplot`, `st.plotly_chart`      |
| Inputs & Forms       | `st.button`, `st.radio`, `st.form`, `st.slider`, etc.          |
| Feedback/Status      | `st.toast`, `st.spinner`, `st.status`, `st.progress`           |
| Media Integration    | `st.image`, `st.audio`, `st.video`                             |
| Custom Extension     | Custom components, 3rd-party components                        |

---

