# Categorical Data in Pandas  

## Overview  
Categorical data in Pandas represents variables with a fixed number of possible values (categories). It is useful for optimizing memory usage and improving performance in operations like filtering and grouping.

## Creating Categorical Data  
### Converting a Column to Categorical  
```python
df["column_name"] = df["column_name"].astype("category")
```
### Creating a Categorical Series  
```python
import pandas as pd  

cat_series = pd.Categorical(["low", "medium", "high", "low"])
```

## Properties of Categorical Data  
- **Categories**: The unique values in the categorical variable.  
- **Ordered**: Whether the categories have a meaningful order (e.g., "low" < "medium" < "high").  
- **Codes**: Numeric representation of categorical values.  

### Accessing Properties  
```python
cat = pd.Categorical(["low", "medium", "high"], categories=["low", "medium", "high"], ordered=True)

print(cat.categories)  # ['low', 'medium', 'high']
print(cat.ordered)  # True
print(cat.codes)  # [0, 1, 2]
```

## Operations on Categorical Data  
### Reordering Categories  
```python
cat = cat.reorder_categories(["high", "medium", "low"], ordered=True)
```
### Adding New Categories  
```python
cat = cat.add_categories(["very high"])
```
### Removing Categories  
```python
cat = cat.remove_categories(["low"])
```
### Renaming Categories  
```python
cat = cat.rename_categories(["L", "M", "H"])
```

## Converting Categorical Data  
### Convert to String  
```python
df["category_column"] = df["category_column"].astype(str)
```
### Convert to Numerical Codes  
```python
df["category_column"] = df["category_column"].cat.codes
```

## Performance Benefits  
- **Memory-efficient**: Uses integer representation instead of strings.  
- **Faster operations**: Improves performance for operations like filtering, grouping, and sorting.  

## Use Cases  
- **Ordinal Data**: Ranking or ordered categories (e.g., ratings: low, medium, high).  
- **Nominal Data**: Unordered categories (e.g., colors, city names).  
- **Reducing Memory Usage**: When working with large categorical datasets.  