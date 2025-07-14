# Django Views Cheatsheet

## 1. Importing Required Modules
- `from django.shortcuts import render, get_object_or_404, redirect`  # Import common shortcuts
- `from django.http import HttpResponse, JsonResponse`  # Import HTTP response classes

## 2. Function-Based Views
- `def view_name(request):`  # Define a function-based view
  - `return HttpResponse('Hello, World!')`  # Return a simple HTTP response

## 3. Rendering Templates
- `def view_name(request):`  # Define a function-based view
  - `context = {'key': 'value'}`  # Define context data
  - `return render(request, 'template.html', context)`  # Render a template with context

## 4. Handling Forms
- `from .forms import FormName`  # Import form class
- `def view_name(request):`  # Define a function-based view
  - `if request.method == 'POST':`  # Check if the request is POST
    - `form = FormName(request.POST)`  # Bind data to the form
    - `if form.is_valid():`  # Validate the form
      - `form.save()`  # Save the form data
      - `return redirect('success_url')`  # Redirect after successful form submission
  - `else:`  # If the request is GET
    - `form = FormName()`  # Instantiate an empty form
  - `return render(request, 'template.html', {'form': form})`  # Render the form in a template

## 5. Using get_object_or_404
- `def view_name(request, id):`  # Define a function-based view with a parameter
  - `obj = get_object_or_404(ModelName, pk=id)`  # Get object or return 404
  - `return render(request, 'template.html', {'object': obj})`  # Render the object in a template

## 6. Redirecting
- `def view_name(request):`  # Define a function-based view
  - `return redirect('url_name')`  # Redirect to another URL

## 7. JSON Responses
- `def view_name(request):`  # Define a function-based view
  - `data = {'key': 'value'}`  # Define data to be returned as JSON
  - `return JsonResponse(data)`  # Return a JSON response

## 8. Class-Based Views
- `from django.views import View`  # Import View class
- `class MyView(View):`  # Define a class-based view
  - `def get(self, request):`  # Handle GET requests
    - `return HttpResponse('Hello, World!')`  # Return a simple HTTP response

## 9. TemplateView
- `from django.views.generic import TemplateView`  # Import TemplateView
- `class MyView(TemplateView):`  # Define a TemplateView
  - `template_name = 'template.html'`  # Specify the template

## 10. ListView
- `from django.views.generic import ListView`  # Import ListView
- `class MyListView(ListView):`  # Define a ListView
  - `model = ModelName`  # Specify the model
  - `template_name = 'template.html'`  # Specify the template
  - `context_object_name = 'objects'`  # Specify the context object name

## 11. DetailView
- `from django.views.generic import DetailView`  # Import DetailView
- `class MyDetailView(DetailView):`  # Define a DetailView
  - `model = ModelName`  # Specify the model
  - `template_name = 'template.html'`  # Specify the template
  - `context_object_name = 'object'`  # Specify the context object name

## 12. CreateView
- `from django.views.generic.edit import CreateView`  # Import CreateView
- `class MyCreateView(CreateView):`  # Define a CreateView
  - `model = ModelName`  # Specify the model
  - `form_class = FormName`  # Specify the form class
  - `template_name = 'template.html'`  # Specify the template
  - `success_url = '/success/'`  # Specify the success URL

## 13. UpdateView
- `from django.views.generic.edit import UpdateView`  # Import UpdateView
- `class MyUpdateView(UpdateView):`  # Define an UpdateView
  - `model = ModelName`  # Specify the model
  - `form_class = FormName`  # Specify the form class
  - `template_name = 'template.html'`  # Specify the template
  - `success_url = '/success/'`  # Specify the success URL

## 14. DeleteView
- `from django.views.generic.edit import DeleteView`  # Import DeleteView
- `class MyDeleteView(DeleteView):`  # Define a DeleteView
  - `model = ModelName`  # Specify the model
  - `template_name = 'template.html'`  # Specify the template
  - `success_url = '/success/'`  # Specify the success URL

## 15. Handling 404 Errors
- `def custom_404_view(request, exception):`  # Define a custom 404 view
  - `return render(request, '404.html', status=404)`  # Render a custom 404 template