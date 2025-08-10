## **user authentication view** (Login, Logout, Register, Password Management) 

---

## **FBV – User Authentication Views**

```python
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect
from django.contrib import messages

# Register View
def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'register.html', {'form': form})

# Login View
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

# Logout View
@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

# Change Password View
@login_required
def change_password_view(request):
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Password changed successfully')
            return redirect('login')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'change_password.html', {'form': form})
```

---

## **CBV – User Authentication Views**

```python
from django.contrib.auth.views import LoginView, LogoutView, PasswordChangeView, PasswordChangeDoneView, PasswordResetView, PasswordResetDoneView, PasswordResetConfirmView, PasswordResetCompleteView
from django.urls import reverse_lazy
from django.views.generic.edit import CreateView
from django.contrib.auth.forms import UserCreationForm

# Register View
class RegisterView(CreateView):
    form_class = UserCreationForm
    template_name = 'register.html'
    success_url = reverse_lazy('login')

# Login View
class UserLoginView(LoginView):
    template_name = 'login.html'

# Logout View
class UserLogoutView(LogoutView):
    next_page = reverse_lazy('login')

# Change Password View
class UserPasswordChangeView(PasswordChangeView):
    template_name = 'change_password.html'
    success_url = reverse_lazy('password_change_done')

class UserPasswordChangeDoneView(PasswordChangeDoneView):
    template_name = 'password_change_done.html'

# Password Reset Views
class UserPasswordResetView(PasswordResetView):
    template_name = 'password_reset.html'
    success_url = reverse_lazy('password_reset_done')

class UserPasswordResetDoneView(PasswordResetDoneView):
    template_name = 'password_reset_done.html'

class UserPasswordResetConfirmView(PasswordResetConfirmView):
    template_name = 'password_reset_confirm.html'
    success_url = reverse_lazy('password_reset_complete')

class UserPasswordResetCompleteView(PasswordResetCompleteView):
    template_name = 'password_reset_complete.html'
```

---

## **URL Mapping Example**

```python
from django.urls import path
from .views import *

urlpatterns = [
    # FBV
    path('register/', register_view, name='register'),
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),
    path('change-password/', change_password_view, name='change_password'),

    # CBV
    path('cbv/register/', RegisterView.as_view(), name='cbv_register'),
    path('cbv/login/', UserLoginView.as_view(), name='cbv_login'),
    path('cbv/logout/', UserLogoutView.as_view(), name='cbv_logout'),
    path('cbv/change-password/', UserPasswordChangeView.as_view(), name='cbv_change_password'),
    path('cbv/change-password/done/', UserPasswordChangeDoneView.as_view(), name='password_change_done'),
]
```

---
