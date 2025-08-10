## **Authentication with CBVs**

### **Login View**

```python
from django.contrib.auth.views import LoginView

class UserLoginView(LoginView):
    template_name = 'auth/login.html'  # Path to login template
    redirect_authenticated_user = True  # Redirect logged-in users
    extra_context = { 'title': 'Login' }  # Additional context
```

---

### **Logout View**

```python
from django.contrib.auth.views import LogoutView

class UserLogoutView(LogoutView):
    next_page = 'login'  # Named URL to redirect after logout
```

---

### **Signup View**

```python
from django.views.generic import CreateView
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy

class UserSignupView(CreateView):
    form_class = UserCreationForm
    template_name = 'auth/signup.html'
    success_url = reverse_lazy('login')
```

---

### **Password Change View**

```python
from django.contrib.auth.views import PasswordChangeView

class UserPasswordChangeView(PasswordChangeView):
    template_name = 'auth/password_change.html'
    success_url = reverse_lazy('password_change_done')
```

---

### **Password Reset Views**

```python
from django.contrib.auth.views import (
    PasswordResetView, PasswordResetConfirmView,
    PasswordResetDoneView, PasswordResetCompleteView
)
from django.urls import reverse_lazy

class UserPasswordResetView(PasswordResetView):
    template_name = 'auth/password_reset.html'
    email_template_name = 'auth/password_reset_email.html'
    success_url = reverse_lazy('password_reset_done')

class UserPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'auth/password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

class UserPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'auth/password_reset_done.html'

class UserPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'auth/password_reset_complete.html'
```

---

### **Restricting Access to Authenticated Users**

```python
from django.contrib.auth.mixins import LoginRequiredMixin

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
    login_url = 'login'  # Named URL for login
    redirect_field_name = 'redirect_to'
```

---

### **Restricting to Specific Permissions**

```python
from django.contrib.auth.mixins import PermissionRequiredMixin

class AdminOnlyView(PermissionRequiredMixin, TemplateView):
    template_name = 'admin_panel.html'
    permission_required = 'auth.view_user'  # Django permission code
```

---

### **Restricting to Specific User Groups**

```python
from django.contrib.auth.mixins import UserPassesTestMixin

class StaffOnlyView(UserPassesTestMixin, TemplateView):
    template_name = 'staff_dashboard.html'

    def test_func(self):
        return self.request.user.is_staff
```

---

### **URL Configuration**

```python
from django.urls import path
from .views import *

urlpatterns = [
    path('login/', UserLoginView.as_view(), name='login'),
    path('logout/', UserLogoutView.as_view(), name='logout'),
    path('signup/', UserSignupView.as_view(), name='signup'),
    path('password_change/', UserPasswordChangeView.as_view(), name='password_change'),
    path('password_reset/', UserPasswordResetView.as_view(), name='password_reset'),
    path('password_reset_confirm/<uidb64>/<token>/', UserPasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password_reset_done/', UserPasswordResetDoneView.as_view(), name='password_reset_done'),
    path('password_reset_complete/', UserPasswordResetCompleteView.as_view(), name='password_reset_complete'),
    path('dashboard/', DashboardView.as_view(), name='dashboard'),
]
```

---
