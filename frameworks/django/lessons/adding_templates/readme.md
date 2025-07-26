## Ways to Add Templates in Django

### 🔹 1. **App-level Template Structure (Default)**

* Templates are placed in each app inside:
  `your_app/templates/your_app_name/`
* Ensures **namespacing** of templates and avoids conflicts.
* Django looks for templates in app directories **if `APP_DIRS=True`** in `TEMPLATES` setting.

---

### 🔹 2. **Global Template Directory (Project-level)**

* A shared folder (e.g., `templates/`) created at the **project root** (same level as `manage.py`).
* Add the path to `DIRS` list in `TEMPLATES` setting:

  ```python
  'DIRS': [BASE_DIR / "templates"],
  ```
* Useful for shared templates like `base.html`, `404.html`, etc.

---

### 🔹 3. **Multiple Global Template Directories**

* You can define **multiple template folders** by adding them in the `DIRS` list:

  ```python
  'DIRS': [
      BASE_DIR / "templates",
      BASE_DIR / "shared_templates",
      BASE_DIR / "external_templates"
  ],
  ```

---

### 🔹 4. **Third-Party Templates (Reusable Apps)**

* If using a third-party app with built-in templates:

  * Keep `APP_DIRS=True`
  * Include the app in `INSTALLED_APPS`
  * Templates are accessed like normal if the app supports it.

---

### 🔹 5. **Custom Template Loaders**

* Advanced usage where you define your own way of loading templates.
* Add custom loaders in `OPTIONS['loaders']`:

  ```python
  'OPTIONS': {
      'loaders': [
          'django.template.loaders.filesystem.Loader',
          'django.template.loaders.app_directories.Loader',
          'your.custom.loader.Class',
      ]
  }
  ```

---

## 🗂 Summary Table

| Method                         | Location                        | When to Use                       |
| ------------------------------ | ------------------------------- | --------------------------------- |
| App-level template folders     | `your_app/templates/your_app/`  | For templates specific to one app |
| Project-level shared folder    | `templates/`                    | For global/shared templates       |
| Multiple custom global folders | Anywhere you define in `DIRS`   | For large/structured projects     |
| Third-party reusable apps      | Inside the installed app folder | For plug-and-play template reuse  |
| Custom template loaders        | Custom Python class             | For advanced/custom logic         |

---
