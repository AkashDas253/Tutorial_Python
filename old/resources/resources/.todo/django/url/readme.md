# Django URL Mapping Cheatsheet

## 1. Importing Required Modules
- `from django.urls import path`  # Import path function
- `from . import views`  # Import views from the current app

## 2. Basic URL Mapping
- `urlpatterns = [path('route/', views.view_name, name='route_name')]`  # Map URL to view

## 3. Including Another URLconf
- `from django.urls import include`  # Import include function
- `urlpatterns = [path('app/', include('app.urls'))]`  # Include another URLconf

## 4. URL Mapping with Parameters
- `urlpatterns = [path('route/<int:id>/', views.view_name, name='route_name')]`  # Map URL with integer parameter
- `urlpatterns = [path('route/<str:name>/', views.view_name, name='route_name')]`  # Map URL with string parameter

## 5. Using re_path for Regular Expressions

- `from django.urls import re_path`  # Import re_path function
- `urlpatterns = [re_path(r'^route/(?P<id>\d+)/$', views.view_name, name='route_name')]`  # Map URL with regex

## 6. Namespacing URL Names

- `app_name = 'app_name'`  # Set application namespace
- `urlpatterns = [path('route/', views.view_name, name='route_name')]`  # Map URL with namespace

## 7. Class-Based Views

- `from django.views.generic import TemplateView`  # Import TemplateView
- `urlpatterns = [path('route/', TemplateView.as_view(template_name='template.html'), name='route_name')]`  # Map URL to class-based view

## 8. Handling 404 Errors

- `handler404 = 'app.views.custom_404_view'`  # Custom 404 error handler

## 9. Handling 500 Errors
- `handler500 = 'app.views.custom_500_view'`  # Custom 500 error handler

## 10. Static Files in Development
- `from django.conf import settings`  # Import settings
- `from django.conf.urls.static import static`  # Import static function
- `urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)`  # Serve static files in development

## 11. Media Files in Development
- `urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)`  # Serve media files in development

## 12. URL Mapping with Default Values
- `urlpatterns = [path('route/<int:id>/', views.view_name, {'default_param': 'value'}, name='route_name')]`  # Map URL with default parameter values

## 13. URL Mapping with Multiple Parameters
- `urlpatterns = [path('route/<int:id>/<str:name>/', views.view_name, name='route_name')]`  # Map URL with multiple parameters

## 14. URL Mapping with Slug
- `urlpatterns = [path('route/<slug:slug>/', views.view_name, name='route_name')]`  # Map URL with slug parameter

## 15. URL Mapping with UUID
- `from uuid import UUID`  # Import UUID
- `urlpatterns = [path('route/<uuid:uuid>/', views.view_name, name='route_name')]`  # Map URL with UUID parameter