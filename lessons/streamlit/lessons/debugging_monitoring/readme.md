## Debugging and Monitoring in Streamlit

Streamlit provides various tools and techniques to **debug issues**, **track performance**, and **monitor app behavior** both during development and in production.

---

## 🐞 Debugging in Streamlit

### 1. **Python Standard Debugging Tools**
Streamlit runs as a Python script, so traditional debugging techniques apply:

| Tool/Method       | Description |
|------------------|-------------|
| `print()`        | Quick output to the terminal or app |
| `st.write()`     | Better than `print()`, logs inline to the UI |
| `logging` module | Standard Python logging |
| `pdb` module     | Command-line step-by-step debugger |

**Example using `st.write`:**
```python
st.write("Debug: variable x =", x)
```

---

### 2. **Using Logging**
Streamlit supports Python logging. Logs appear in the terminal.

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug("Debug info")
```

**Levels:**
- `DEBUG`: Detailed info for diagnostics
- `INFO`: General app status
- `WARNING`, `ERROR`, `CRITICAL`: For problems

---

### 3. **Exceptions and Tracebacks**
Streamlit displays full stack traces in the browser when exceptions occur.

| Feature          | Description |
|------------------|-------------|
| Auto traceback   | Error messages shown inline with stacktrace |
| Developer alerts | Shows the error and points to line number |
| Tip              | Avoid hiding exceptions in `try-except` during dev |

---

### 4. **Streamlit Runtime Behavior**
| Behavior        | Description |
|-----------------|-------------|
| Reruns app on every change in widget value |
| Reruns on file save (`.py`) automatically |
| Use `st.experimental_rerun()` to force rerun |

---

## 📊 Monitoring Performance

### 1. **Execution Time Tracking**
Use Python `time` module or performance profiling tools.

**Basic Example:**
```python
import time
start = time.time()
# ...code...
st.write("Execution time:", time.time() - start)
```

---

### 2. **Profiling with `cProfile`**
For deeper insight into performance bottlenecks.

```bash
python -m cProfile -s time your_app.py
```

---

### 3. **Streamlit Profiler Plugin**
Third-party tool:
- Install: `pip install streamlit-profiler`
- Usage:
```python
from streamlit_profiler import Profiler

with Profiler():
    your_function()
```

---

## 📈 Monitoring Deployed Apps

### 1. **Streamlit Community Cloud**
Provides:
- **Error logs**
- **App health metrics**
- **Resource usage (memory, CPU)**

Available in the **"Logs"** tab of deployed apps.

---

### 2. **External Monitoring Tools**
| Tool            | Integration Type |
|------------------|------------------|
| Prometheus       | Use with a wrapper for metrics |
| Grafana          | For dashboarding logs/metrics |
| Sentry           | Error monitoring (manual integration) |
| Google Analytics | Page tracking via extensions |
| Logging Services | Log to files or cloud (e.g., AWS CloudWatch) |

---

### 3. **Tracking Session and User Behavior**
Use:
- `st.session_state`
- `st.experimental_get_query_params()` to capture URL params
- Custom JS in components to track client behavior

---

## 🧪 Testing and CI

| Tool         | Description |
|--------------|-------------|
| `pytest`     | Standard unit testing |
| `streamlit.testing` (planned) | Future module for testable UI units |
| GitHub Actions | For CI pipelines, deploy on push |
| Docker        | Monitor and debug containerized apps |

---

## 🚨 Common Debugging Scenarios

| Issue                      | Solution |
|----------------------------|----------|
| App reruns unexpectedly    | Use `st.session_state` properly |
| Data not persisting        | Use caching (`@st.cache_data`) or session state |
| API calls repeated         | Use caching and store results in state |
| Long execution blocks UI   | Use `st.spinner()` and async calls |
| Component not rendering    | Check HTML/JS structure, browser console |
| No logs on production      | Enable logs or connect to log monitoring tool |

---
