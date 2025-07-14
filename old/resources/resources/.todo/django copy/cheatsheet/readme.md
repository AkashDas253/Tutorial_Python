# Django Cheatsheet

## 1. Installing Django
- pip install django  ### Install Django

## 2. Creating a New Project
- django-admin startproject project_name  # Create a new Django project

## 3. Running the Development Server
- python manage.py runserver  # Start the development server

## 4. Creating a New App
- python manage.py startapp app_name  # Create a new Django app

## 5. Defining URL Patterns
- from django.urls import path  # Import path
- urlpatterns = [
  - path('home/', views.home, name='home'),  # Define URL pattern
] 

## 6. Creating Views
- from django.shortcuts import render  # Import render
- def home(request):
  - return render(request, 'home.html')  # Render HTML template

## 7. Using Templates
- INSTALLED_APPS = [
  - 'app_name',  # Add app to INSTALLED_APPS in settings.py
]
- {% extends 'base.html' %}  # Template inheritance
- {% block content %}  # Block definition

## 8. Creating Models
- from django.db import models  # Import models
- class MyModel(models.Model):
  - field_name = models.CharField(max_length=100)  # Define model field

## 9. Making Migrations
- python manage.py makemigrations  # Create migration files
- python manage.py migrate  # Apply migrations to the database

## 10. Using the Django Admin
- from django.contrib import admin  # Import admin
- admin.site.register(MyModel)  # Register model with admin

## 11. Accessing the Database
- from app_name.models import MyModel  # Import model
- MyModel.objects.all()  # Get all objects
- MyModel.objects.filter(field_name='value')  # Filter objects

## 12. Using Forms
- from django import forms  # Import forms
- class MyForm(forms.Form):
  - field_name = forms.CharField(max_length=100)  # Define form field

## 13. Handling Form Submission
- if request.method == 'POST':
  - form = MyForm(request.POST)  # Create form instance with POST data
  - if form.is_valid():
    - data = form.cleaned_data  # Access cleaned data

## 14. Using Middleware
- MIDDLEWARE = [
  - 'django.middleware.security.SecurityMiddleware',  # Example middleware
]

## 15. Customizing Settings
- DEBUG = True  # Enable debug mode
- ALLOWED_HOSTS = ['localhost', '127.0.0.1']  # Allowed hosts for deployment
