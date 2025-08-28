## Django Forms 

### Purpose

* Handle user input from HTML forms securely and efficiently.
* Validate and clean data before saving to the database.
* Generate form HTML automatically from Python code.

---

### Types of Forms

* **`forms.Form`** – Manual field declaration for custom forms (not tied to a model).
* **`forms.ModelForm`** – Auto-generates fields from a model, linking form directly to database structure.

---

### Workflow of Forms in Django

1. **Define the Form**

   * Use `forms.Form` or `forms.ModelForm`.
   * Specify fields and validation rules.
2. **Display the Form in a View**

   * Instantiate form object.
   * Pass it to the template for rendering.
3. **Handle Submission**

   * Check `request.method` for `POST`.
   * Bind `request.POST` (and `request.FILES` if file upload) to the form.
   * Call `is_valid()` to run validations.
4. **Access Cleaned Data**

   * Use `form.cleaned_data` after validation.
5. **Save Data**

   * For `ModelForm`, use `.save()` to create/update model instances.

---

### Key Attributes & Methods

* `is_valid()` – Runs validation checks and returns boolean.
* `clean_<fieldname>()` – Custom validation for individual fields.
* `clean()` – Cross-field validation logic.
* `errors` – Returns dictionary of validation errors.
* `as_p()`, `as_table()`, `as_ul()` – Render form as HTML.
* `Meta` class in `ModelForm` – Defines model and fields to include/exclude.

---

### Field Types (Common Examples)

* `CharField`, `EmailField`, `IntegerField`, `BooleanField`, `ChoiceField`, `DateField`, `FileField`, `ImageField`.
* Each can have attributes like `required`, `max_length`, `initial`, `widget`, `validators`.

---

### Widgets

* Define HTML rendering for a field (e.g., `TextInput`, `Textarea`, `CheckboxInput`, `Select`).
* Can be customized to add CSS classes, placeholders, etc.

---

### Form Validation Flow

1. Instantiate form with data.
2. Call `.is_valid()` → triggers field validators and custom `clean()` methods.
3. Errors stored in `.errors` if any.
4. Valid data available in `.cleaned_data`.

---

### File Uploads in Forms

* Use `enctype="multipart/form-data"` in HTML form.
* Access uploaded files via `request.FILES`.

---

### Security in Forms

* CSRF protection is enabled by default with `{% csrf_token %}` in templates.
* Prevents malicious cross-site request submissions.

---

### Usage Scenarios

* Contact forms, login forms, signup forms.
* Profile update forms linked to a user model.
* Data filtering forms in admin or front-end dashboards.

---
