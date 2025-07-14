# URLs

## Setting up
- URL configurations in Django.

## URL Configuration

### Project-Level URLs
- Defines URL patterns in the project's `urls.py`.
- Routes URLs to views at the project level.

  ```python
  # project/urls.py
  from django.contrib import admin
  from django.urls import path, include

  urlpatterns = [
      path('admin/', admin.site.urls),
      path('app_name/', include('app_name.urls')),
  ]
  ```

### App-Level URLs
- Defines URL patterns in the app's `urls.py`.
- Routes URLs to views at the app level.

  ```python
  # app_name/urls.py
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.index, name='index'),
  ]
  ```

### Including URLs
- Includes app-level URLs in the project-level `urls.py`.

  ```python
  # project/urls.py
  from django.contrib import admin
  from django.urls import path, include

  urlpatterns = [
      path('admin/', admin.site.urls),
      path('app_name/', include('app_name.urls')),
  ]
  ```

## URL Patterns

### Path Function
- Uses the `path` function to define URL patterns.
- Simple string-based URL patterns.

  ```python
  # project/urls.py
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.home, name='home'),
      path('about/', views.about, name='about'),
  ]
  ```

### Re_path Function
- Uses the `re_path` function for regex-based URL patterns.
- Allows complex URL matching using regular expressions.

  ```python
  # project/urls.py
  from django.urls import re_path
  from . import views

  urlpatterns = [
      re_path(r'^articles/(?P<year>[0-9]{4})/$', views.year_archive, name='year_archive'),
  ]
  ```

### URL Parameters
- Captures URL parameters and passes them to views.
- Supports different types like `<int:id>`, `<str:name>`, etc.

  ```python
  # project/urls.py
  from django.urls import path
  from . import views

  urlpatterns = [
      path('article/<int:id>/', views.article_detail, name='article_detail'),
  ]
  ```

### Named URL Patterns
- Names URL patterns for easier reference.
- Useful for reverse URL resolution.

  ```python
  # project/urls.py
  from django.urls import path
  from . import views

  urlpatterns = [
      path('', views.home, name='home'),
      path('about/', views.about, name='about'),
  ]
  ```

## URL Routing

### Static Files
- Serves static files during development.
- Uses `static()` to append static URL patterns.

  ```python
  # settings.py
  from django.conf import settings
  from django.conf.urls.static import static

  urlpatterns = [
      # ... your url patterns
  ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
  ```

### Media Files
- Serves media files during development.
- Uses `static()` to append media URL patterns.

  ```python
  # settings.py
  from django.conf import settings
  from django.conf.urls.static import static

  urlpatterns = [
      # ... your url patterns
  ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
  ```

## URL Namespaces

### Using Namespaces
- Uses namespaces to organize URL names.
- Helps avoid URL name conflicts.

  ```python
  # project/urls.py
  from django.urls import path, include

  urlpatterns = [
      path('app_name/', include(('app_name.urls', 'app_name'), namespace='app_name')),
  ]
  ```

### Including Namespaces
- Includes namespaces in URL configurations.
- Defines `app_name` in the app's `urls.py`.

  ```python
  # app_name/urls.py
  from django.urls import path
  from . import views

  app_name = 'app_name'

  urlpatterns = [
      path('', views.index, name='index'),
  ]
  ```

## URL Reverse Resolution

### Using reverse()
- Uses the `reverse()` function to get URL paths from view names.
- Useful for generating URLs dynamically.

  ```python
  from django.urls import reverse

  def my_view(request):
      url = reverse('app_name:index')
      # use the url
  ```

### Using reverse_lazy()
- Uses the `reverse_lazy()` function for URL resolution.
- Useful in class-based views where URL resolution is needed before the view is fully initialized.

  ```python
  from django.urls import reverse_lazy
  from django.views.generic.edit import CreateView
  from .models import MyModel

  class MyCreateView(CreateView):
      model = MyModel
      success_url = reverse_lazy('app_name:index')
  ```