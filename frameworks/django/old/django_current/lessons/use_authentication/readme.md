## **Using Authentication in Django**

### **Authentication System Overview**

* **Authentication**: Verifies the identity of a user (login/logout).
* **Authorization**: Determines what actions a user can perform (permissions, groups).
* Built into `django.contrib.auth`.

---

### **Core Components**

* **User Model**

  * Default: `django.contrib.auth.models.User`
  * Custom: Extend `AbstractUser` or `AbstractBaseUser`
* **Authentication Backends**

  * Defines how Django authenticates (e.g., `ModelBackend`).
* **Sessions**

  * Store user authentication state in cookies.

---

### **Key Functions (django.contrib.auth)**

| Function                                              | Purpose                                       |
| ----------------------------------------------------- | --------------------------------------------- |
| `authenticate(request, username=None, password=None)` | Verify credentials; returns `User` or `None`. |
| `login(request, user)`                                | Persist session for authenticated user.       |
| `logout(request)`                                     | Clear session data.                           |
| `get_user(request)`                                   | Get currently authenticated user.             |

---

### **Decorators and Mixins**

* **Function-Based Views (FBV)**

  * `@login_required` → Redirects anonymous users to login page.
* **Class-Based Views (CBV)**

  * `LoginRequiredMixin`
  * `PermissionRequiredMixin`
  * `UserPassesTestMixin`

---

### **Authentication Views**

Django provides ready-made views in `django.contrib.auth.views`:

| View                        | Purpose                      | URL Name                  |
| --------------------------- | ---------------------------- | ------------------------- |
| `LoginView`                 | Login form handling          | `login`                   |
| `LogoutView`                | Logs out user                | `logout`                  |
| `PasswordChangeView`        | Change password              | `password_change`         |
| `PasswordChangeDoneView`    | Confirm password change      | `password_change_done`    |
| `PasswordResetView`         | Request password reset       | `password_reset`          |
| `PasswordResetDoneView`     | Reset request confirmation   | `password_reset_done`     |
| `PasswordResetConfirmView`  | Reset password link handling | `password_reset_confirm`  |
| `PasswordResetCompleteView` | Password reset complete page | `password_reset_complete` |

---

### **Settings for Authentication**

```python
# settings.py
LOGIN_URL = '/login/'  # Redirect target for @login_required
LOGIN_REDIRECT_URL = '/'  # After successful login
LOGOUT_REDIRECT_URL = '/'  # After logout
AUTH_USER_MODEL = 'myapp.CustomUser'  # Custom user model (if used)
```

---

### **Forms for Authentication**

* Built-in forms from `django.contrib.auth.forms`:

  * `AuthenticationForm` → Login
  * `UserCreationForm` → Registration
  * `UserChangeForm` → Profile update
  * `PasswordChangeForm`
  * `PasswordResetForm`
  * `SetPasswordForm`

---

### **Usage Scenarios**

#### **Login**

```python
from django.contrib.auth import authenticate, login

user = authenticate(request, username='john', password='secret')
if user is not None:
    login(request, user)
```

#### **Logout**

```python
from django.contrib.auth import logout

logout(request)
```

#### **Check Authentication Status**

```python
if request.user.is_authenticated:
    # Authenticated user
else:
    # Guest
```

#### **Check Permissions**

```python
if request.user.has_perm('app_name.permission_code'):
    # Allowed
```

#### **Groups**

```python
from django.contrib.auth.models import Group
group = Group.objects.get(name='Editors')
user.groups.add(group)
```

---

### **Template Tags**

```django
{% if user.is_authenticated %}
    Hello, {{ user.username }}
{% else %}
    <a href="{% url 'login' %}">Login</a>
{% endif %}
```

---
