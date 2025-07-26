## ⚙️ Virtual Environment Setup on **Windows (PowerShell)** for Django

### ✅ Step-by-step Commands

```powershell
# Step 1: Create a virtual environment named .venv
python -m venv .venv
```

* Creates a folder named `.venv` in the current directory.
* All dependencies will be isolated inside `.venv`.

---

```powershell
# Step 2: (Optional but recommended) Allow script execution for virtual envs
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

* Required to run `Activate.ps1` scripts in PowerShell.
* `RemoteSigned` allows local scripts to run without signing.

---

```powershell
# Step 3: Activate the virtual environment
.venv\Scripts\Activate.ps1
```

* Activates the `.venv` so that `python` and `pip` refer to the virtual environment.

> 🔹 After this, your prompt will change to show `(.venv)` at the beginning, indicating the environment is active.

---

### ✅ Step 4: Install Django

```powershell
pip install django
```

---

### ✅ Step 5: Save dependencies

```powershell
pip freeze > requirements.txt
```

---

### ✅ Step 6: Add to `.gitignore`

```
.venv/
```

---

### 🔹 To Deactivate

```powershell
deactivate
```

---

### 🧩 Alternative Activation Methods

| Shell      | Command                         |
| ---------- | ------------------------------- |
| PowerShell | `.venv\Scripts\Activate.ps1`    |
| CMD        | `.venv\Scripts\activate.bat`    |
| Git Bash   | `source .venv/Scripts/activate` |

---
