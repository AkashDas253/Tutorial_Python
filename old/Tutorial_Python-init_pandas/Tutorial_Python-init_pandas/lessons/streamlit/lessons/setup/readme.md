## **Streamlit Installation and Setup**

Streamlit is easy to install and requires minimal setup. It can be installed on various operating systems and environments. Below is a step-by-step guide.

---

### **1. Prerequisites**
| Requirement       | Description                                                   |
|--------------------|---------------------------------------------------------------|
| Python Version     | Requires Python **3.7 - 3.11**.                              |
| Package Manager    | `pip` (preferred for installation).                          |
| Operating System   | Compatible with Windows, macOS, and Linux.                   |
| Virtual Environment| Recommended to isolate dependencies using `venv` or `conda`. |

---

### **2. Installation Steps**

#### **a. Installing Streamlit**
Run the following command to install Streamlit using `pip`:

```bash
pip install streamlit
```

#### **b. Verifying Installation**
To confirm that Streamlit is installed correctly, check the version:

```bash
streamlit --version
```

#### **c. Running Your First Streamlit App**
1. Create a Python file, e.g., `app.py`.
2. Add a simple Streamlit command, such as:
   ```python
   import streamlit as st
   st.title("Hello, Streamlit!")
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

### **3. Configuration Options**

Streamlit uses a configuration file (`~/.streamlit/config.toml`) to customize the behavior of apps. Below is a detailed table of configuration parameters:

| **Category**         | **Parameter**            | **Default Value**  | **Description**                                                 |
|-----------------------|--------------------------|---------------------|-----------------------------------------------------------------|
| **Server**           | `headless`              | `true`             | Runs Streamlit in headless mode (no UI for the local server).   |
|                       | `port`                  | `8501`             | The port where the app will run.                               |
|                       | `enableCORS`            | `true`             | Prevents cross-origin requests unless disabled.                |
|                       | `serverAddress`         | `localhost`        | IP address for the server.                                     |
| **Browser**          | `gatherUsageStats`       | `true`             | Sends anonymized usage stats to Streamlit developers.          |
|                       | `server.maxUploadSize`  | `200` (MB)         | Maximum file upload size allowed.                              |
| **Caching**          | `cache.enable`          | `true`             | Enables or disables caching.                                   |
|                       | `cache.ttl`             | `None`             | Time-to-live for cached data (in seconds).                     |
| **Theme**            | `theme.primaryColor`     | `#F63366`          | Changes the app's primary color.                               |
|                       | `theme.backgroundColor` | `#FFFFFF`          | Sets the app background color.                                 |

---

### **4. Updating Streamlit**
To update Streamlit to the latest version, use:

```bash
pip install --upgrade streamlit
```

---

### **5. Troubleshooting Common Issues**
| **Issue**                     | **Solution**                                                                 |
|--------------------------------|-------------------------------------------------------------------------------|
| Command not found: `streamlit` | Ensure Python and pip are added to your PATH environment variable.            |
| Dependency conflicts           | Use a virtual environment to isolate your project dependencies.              |
| Port already in use            | Specify a different port using `--server.port`, e.g., `streamlit run app.py --server.port=8502`. |
| Permissions error              | Use `pip install --user streamlit` or run as an administrator.                |

---

### **6. Advanced Setup**

#### **a. Installing in a Virtual Environment**
1. Create a virtual environment:
   ```bash
   python -m venv streamlit-env
   ```
2. Activate the environment:
   - **Windows**: `streamlit-env\Scripts\activate`
   - **macOS/Linux**: `source streamlit-env/bin/activate`
3. Install Streamlit:
   ```bash
   pip install streamlit
   ```

#### **b. Running Streamlit with Specific Options**
Streamlit apps can be launched with additional server and debugging options:

| **Option**                | **Command**                                      | **Description**                                           |
|---------------------------|--------------------------------------------------|-----------------------------------------------------------|
| Specifying port           | `streamlit run app.py --server.port=8502`        | Runs the app on port 8502.                                |
| Debug mode                | `streamlit run app.py --global.logLevel debug`   | Enables detailed logs for debugging.                      |
| Headless mode             | `streamlit run app.py --server.headless=true`    | Runs the app without opening a browser.                   |

---

### **7. Uninstalling Streamlit**
To remove Streamlit, use:

```bash
pip uninstall streamlit
```

---

### **8. Example Configuration File**
You can manually customize settings by editing the `config.toml` file located in `~/.streamlit/`. Below is an example:

```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1E90FF"
backgroundColor = "#F0F0F5"
```

---
