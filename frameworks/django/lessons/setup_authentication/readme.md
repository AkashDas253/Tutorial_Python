## Setting Up User Authentication in Django

### Enable Authentication System

* Django's built-in authentication system is available in `django.contrib.auth` and `django.contrib.contenttypes`.
* Ensure they are listed in `INSTALLED_APPS` in `settings.py`.

---

### Configure Authentication Settings

* `AUTH_USER_MODEL` → Specify custom user model if overriding.
* `LOGIN_URL` → Path for login page if user tries accessing protected views.
* `LOGIN_REDIRECT_URL` → Path to redirect after successful login.
* `LOGOUT_REDIRECT_URL` → Path to redirect after logout.

---

### Create Authentication URLs (`urls.py`)

```python
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),  
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  
]
```

---

### Protect Views with Login Requirement

```python
from django.contrib.auth.decorators import login_required

@login_required
def dashboard(request):
    return render(request, 'dashboard.html')
```

For CBV:

```python
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView

class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = 'dashboard.html'
```

---

### Using `UserCreationForm` for Registration

```python
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})
```

---

### Template Considerations

* `login.html` should have form fields for `username` and `password`.
* Use `{% csrf_token %}` inside forms.
* Access logged-in user with `{{ user.username }}` in templates.

---
