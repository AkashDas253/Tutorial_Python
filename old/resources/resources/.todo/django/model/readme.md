# Django Models Cheatsheet

## 1. Importing Models
- `from django.db import models`  # Import models module

## 2. Defining a Model
- `class ModelName(models.Model):`  # Define a new model
  - `field_name = models.FieldType()`  # Define model fields

## 3. Common Field Types
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

## 4. Field Options
- `null=True`  # Allow NULL values
- `blank=True`  # Allow blank values
- `default=value`  # Set default value
- `unique=True`  # Ensure unique values
- `choices=[(value1, 'Label1'), (value2, 'Label2')]`  # Define choices for the field

## 5. Primary Key
- `id = models.AutoField(primary_key=True)`  # Define a primary key (automatically added by Django)

## 6. ForeignKey
- `foreign_key = models.ForeignKey('OtherModel', on_delete=models.CASCADE)`  # Define a foreign key relationship

## 7. OneToOneField
- `one_to_one = models.OneToOneField('OtherModel', on_delete=models.CASCADE)`  # Define a one-to-one relationship

## 8. ManyToManyField
- `many_to_many = models.ManyToManyField('OtherModel')`  # Define a many-to-many relationship

## 9. Meta Options
- `class Meta:`  # Define meta options for the model
  - `ordering = ['field_name']`  # Default ordering
  - `verbose_name = 'Verbose Name'`  # Human-readable name
  - `verbose_name_plural = 'Verbose Names'`  # Human-readable plural name

## 10. String Representation
- `def __str__(self):`  # Define string representation of the model
  - `return self.field_name`  # Return a string

## 11. Custom Model Methods
- `def custom_method(self):`  # Define a custom method for the model
  - `return some_value`  # Return a value

## 12. Model Inheritance
- `class ChildModel(ParentModel):`  # Define a model that inherits from another model

## 13. Abstract Base Classes
- `class AbstractModel(models.Model):`  # Define an abstract base class
  - `class Meta:`  # Define meta options
    - `abstract = True`  # Mark the model as abstract

## 14. Proxy Models
- `class ProxyModel(OriginalModel):`  # Define a proxy model
  - `class Meta:`  # Define meta options
    - `proxy = True`  # Mark the model as a proxy

## 15. Model Managers
- `class CustomManager(models.Manager):`  # Define a custom manager
  - `def custom_method(self):`  # Define a custom method for the manager
    - `return self.filter(some_filter)`  # Return a queryset

- `objects = CustomManager()`  # Use the custom manager in the model

## 16. Signals
- `from django.db.models.signals import post_save`  # Import signals
- `from django.dispatch import receiver`  # Import receiver decorator

- `@receiver(post_save, sender=ModelName)`  # Define a signal receiver
  - `def post_save_handler(sender, instance, **kwargs):`  # Define the handler function
    - `# Do something after the model is saved`

## 17. Making Migrations
- `python manage.py makemigrations`  # Create new migrations based on model changes
- `python manage.py migrate`  # Apply migrations to the database