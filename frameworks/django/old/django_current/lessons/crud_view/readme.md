## CRUD Views in Django

### Overview

CRUD views handle the four main operations for database objects: **Create**, **Read**, **Update**, and **Delete**. Django can implement these with **function-based views (FBVs)** or **class-based generic views (CBVs)**.

---

### CRUD Operations

* **Create**

  * Display a form to create a new record
  * Handle form submission and save data to the database
  * Views: `CreateView` (CBV) or manual form handling in FBV

* **Read**

  * Display data from the database
  * Can be a list of objects (`ListView`) or a single object's details (`DetailView`)

* **Update**

  * Display a pre-filled form for an existing object
  * Save changes to the database
  * Views: `UpdateView` (CBV) or manual update handling in FBV

* **Delete**

  * Confirm and delete a record
  * Views: `DeleteView` (CBV) or manual deletion in FBV

---

### Function-Based View (FBV) Approach

* Explicitly handle **GET** and **POST** requests
* Manually bind data to forms and validate
* More control but more code to write

---

### Class-Based View (CBV) Approach

* Use Django’s generic views:

  * `ListView` → Read multiple
  * `DetailView` → Read single
  * `CreateView` → Create
  * `UpdateView` → Update
  * `DeleteView` → Delete
* Requires specifying `model`, `fields`, and `template_name`

---

### URL Mapping for CRUD

* Define named paths for each CRUD action
* Example naming:

  * `"object_list"` → Read multiple
  * `"object_detail"` → Read single
  * `"object_create"` → Create
  * `"object_update"` → Update
  * `"object_delete"` → Delete

---

### Template Structure

* Separate templates for list, detail, form, and delete confirmation
* Use Django’s template tags for form rendering and CSRF protection

---
