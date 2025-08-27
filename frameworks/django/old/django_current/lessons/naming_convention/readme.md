## **Naming Conventions in Django**

### **General Python Conventions (PEP 8)**

* Follow **PEP 8** for consistency across the Django project.
* Use **lowercase\_with\_underscores** for functions, variables, and file names.
* Use **CapWords (PascalCase)** for class names.
* Constants should be in **UPPERCASE\_WITH\_UNDERSCORES**.

---

### **Project & App Naming**

* Project and app names should be **short, descriptive, lowercase** without spaces.
* Avoid special characters; underscores allowed but discouraged for app names.
* Example:

  * ✅ `blog`
  * ❌ `BlogApp` (for app name)

---

### **Model Naming**

* Class names: **PascalCase**, singular form.

  * ✅ `Article`, `UserProfile`
  * ❌ `Articles`, `userprofile`
* Related database table names: automatically lowercase with app label prefix (can override with `db_table` in `Meta`).

---

### **Field Naming**

* Use lowercase\_with\_underscores.
* Name should describe the data.

  * ✅ `first_name`, `created_at`
  * ❌ `fn`, `ca`

---

### **View Naming**

* Function-based views: lowercase\_with\_underscores.
* Class-based views: PascalCase with `View` suffix.

  * ✅ `ArticleListView`, `create_article`
  * ❌ `articlelist`, `CreateArticle`

---

### **Template Naming**

* Use lowercase\_with\_underscores.
* Group by app, e.g.:

  * `templates/blog/article_list.html`
  * `templates/blog/article_detail.html`

---

### **URL Naming**

* URL patterns: lowercase\_with\_underscores.
* Named URLs: descriptive, use hyphen for readability in URL paths, underscore in `name`.

  * Path: `/blog/article-list/`
  * Name: `article_list`

---

### **Static & Media Files**

* Folder names: lowercase\_with\_underscores.
* File names: lowercase\_with\_underscores.

  * Example: `static/blog/css/styles.css`

---

### **Admin Naming**

* Class names: PascalCase with `Admin` suffix.

  * ✅ `ArticleAdmin`
* Variable names for registration: lowercase.

  * ✅ `admin.site.register(Article, ArticleAdmin)`

---

### **Test Naming**

* Test classes: PascalCase starting with `Test`.
* Test functions: lowercase\_with\_underscores describing behavior.

  * ✅ `TestArticleModel`, `test_article_creation`

---
