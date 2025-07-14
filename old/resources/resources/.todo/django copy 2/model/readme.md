# Model

## Model Operations

### Creating Models
- Defines models in `models.py`.

#### Syntax:
  ```python
  # app/models.py
  from django.db import models # Import models module

  class ModelName(models.Model):  # Define a new model
    field_name = models.FieldType()  # Define model fields
      ...
    class Meta:
      key = value
      ...
    def function(self): 
      ...
      return specific value
    def __str__(self):
      ...
      return string_representation # Return string representation of object
  ```

### Common Field Types
- `CharField(max_length=100)`  # String field with max length
- `TextField()`  # Large text field
- `IntegerField()`  # Integer field
- `FloatField()`  # Float field
- `BooleanField()`  # Boolean field
- `DateField()`  # Date field
- `DateTimeField()`  # Date and time field
- `EmailField()`  # Email field
- `URLField()`  # URL field
- `FileField(upload_to='uploads/')`  # File upload field
- `ImageField(upload_to='images/')`  # Image upload field

### Field Options
- `null=True`  # Allow NULL values
- `blank=True`  # Allow blank values
- `default=value`  # Set default value
- `unique=True`  # Ensure unique values
- `choices=[(value1, 'Label1'), (value2, 'Label2')]`  # Define choices for the field

### Relation Field:

- `id = models.AutoField(primary_key=True)`  # Define a primary key (automatically added by Django)
- `foreign_key = models.ForeignKey('OtherModel', on_delete=models.CASCADE)`  # Define a foreign key relationship
- `one_to_one = models.OneToOneField('OtherModel', on_delete=models.CASCADE)`  # Define a one-to-one relationship
- `many_to_many = models.ManyToManyField('OtherModel')`  # Define a many-to-many relationship

### Model Methods
- Custom methods on models.

  ```python
  from django.db import models

  class MyModel(models.Model): 
      name = models.CharField(max_length=100)
      age = models.IntegerField()

      def greet(self):
          return f"Hello, my name is {self.name}"
  ```

### Model Metadata
- Uses `Meta` class for model metadata.

  ```python
  from django.db import models

  class MyModel(models.Model):
      name = models.CharField(max_length=100)

      class Meta:
          ordering = ['name']
          verbose_name = "My Model"
  ```

### Model Managers
- Uses and creates custom model managers.

  ```python
  from django.db import models

  class MyModelManager(models.Manager):
      def active(self):
          return self.filter(is_active=True)

  class MyModel(models.Model):
      name = models.CharField(max_length=100)
      is_active = models.BooleanField(default=True)

      objects = MyModelManager()
  ```

### Model Relationships
- Defines relationships between models (OneToOne, ForeignKey, ManyToMany).

  ```python
  from django.db import models

  class Author(models.Model):
      name = models.CharField(max_length=100)

  class Book(models.Model):
      title = models.CharField(max_length=100)
      author = models.ForeignKey(Author, on_delete=models.CASCADE)  # ForeignKey relationship
  ```

## Model Migrations

### Creating Migrations
- Creates migrations using `makemigrations`.

  ```sh
  python manage.py makemigrations
  ```

### Applying Migrations
- Applies migrations using `migrate`.

  ```sh
  python manage.py migrate
  ```

### Rolling Back Migrations
- Rolls back migrations.

  ```sh
  python manage.py migrate app_name 0001
  ```

## Model Queries

### QuerySet API
- Uses the QuerySet API for querying the database.

  ```python
  from myapp.models import MyModel

  queryset = MyModel.objects.all()
  ```

### Filtering Queries
- Filters query results.

  ```python
  queryset = MyModel.objects.filter(name='John')
  ```

### Aggregation and Annotation
- Uses aggregation and annotation in queries.

  ```python
  from django.db.models import Count

  queryset = MyModel.objects.annotate(num_books=Count('book'))
  ```

### Query Optimization
- Tips for optimizing database queries.

  ```python
  queryset = MyModel.objects.select_related('author').all()
  ```

## Model Serialization

### Serializing Models
- Serializes models to JSON or other formats.

  ```python
  from django.core.serializers import serialize

  data = serialize('json', MyModel.objects.all())
  ```

### Deserializing Data
- Deserializes data into models.

  ```python
  from django.core.serializers import deserialize

  for obj in deserialize('json', data):
      obj.save()
  ```

## Model Forms

### Creating Model Forms
- Creates forms from models using `ModelForm`.

  ```python
  from django import forms
  from .models import MyModel

  class MyModelForm(forms.ModelForm):
      class Meta:
          model = MyModel
          fields = ['name', 'age']
  ```

### Validating Model Forms
- Validates model forms.

  ```python
  form = MyModelForm(data={'name': 'John', 'age': 30})
  if form.is_valid():
      # Process form data
      pass
  ```

### Saving Model Forms
- Saves data from model forms.

  ```python
  form = MyModelForm(data={'name': 'John', 'age': 30})
  if form.is_valid():
      form.save()
  ```

## Model Admin

### Registering Models with Admin
- Registers models with the Django admin site.

  ```python
  from django.contrib import admin
  from .models import MyModel

  admin.site.register(MyModel)
  ```

### Customizing Admin Interface
- Customizes the admin interface for models.

  ```python
  from django.contrib import admin
  from .models import MyModel

  class MyModelAdmin(admin.ModelAdmin):
      list_display = ('name', 'age')

  admin.site.register(MyModel, MyModelAdmin)
  ```

### Admin Actions
- Defines custom actions in the admin interface.

  ```python
  from django.contrib import admin
  from .models import MyModel

  def make_active(modeladmin, request, queryset):
      queryset.update(is_active=True)

  class MyModelAdmin(admin.ModelAdmin):
      actions = [make_active]

  admin.site.register(MyModel, MyModelAdmin)
  ```