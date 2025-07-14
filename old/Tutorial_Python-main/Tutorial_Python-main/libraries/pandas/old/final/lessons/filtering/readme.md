
# **Filtering in pandas**

---

## **Definition**
Filtering in pandas refers to extracting specific rows or columns from a DataFrame or Series based on one or more conditions. It enables data subsetting and conditional selection.

---

## **Types of Filtering**

### **Boolean Indexing**
Filter rows based on a condition that returns `True` or `False`.

```python
df[df['column'] > 10]
df[df['column'] == 'value']
df[df['column'].isna()]
df[(df['A'] > 5) & (df['B'] < 20)]
```

---

### **Using `.query()`**
SQL-like syntax to filter rows.

```python
df.query("column > 10")
df.query("A == 'foo' and B < 5")
```

- Only works with column names that are valid identifiers.
- Faster on large datasets.
- Can use `@` to refer to local variables:
  
```python
value = 10
df.query("column > @value")
```

---

### **Using `.where()` and `.mask()`**
Filter conditionally while retaining the original shape (fills False values with NaN).

```python
df.where(df['column'] > 10)      # Keeps only where condition is True
df.mask(df['column'] > 10)       # Keeps only where condition is False
```

- Returns a DataFrame with same shape.
- Can pass `other=value` to replace.

---

### **Using `.isin()`**
Check membership in a list or Series.

```python
df[df['column'].isin([1, 2, 3])]
df[~df['column'].isin(['a', 'b'])]  # NOT in
```

---

### **Filtering with `.str` and `.dt` accessors**
For filtering strings or datetime columns.

```python
df[df['name'].str.startswith('A')]
df[df['date'].dt.year == 2024]
```

---

### **Filter Columns or Index**
Use `.filter()` for column or row filtering by name/label.

```python
df.filter(items=['col1', 'col2'])                 # Exact match
df.filter(like='202', axis=1)                     # Substring match in column names
df.filter(regex='^Q[1-4]', axis=0)                # Regex on row index
```

---

## **Negating Conditions**
Use `~` for negation.

```python
df[~(df['column'] > 10)]
```

---

## **Lambda-Based Filtering**
Apply `lambda` functions inside `.loc[]` or `.apply()`.

```python
df.loc[lambda x: x['column'] > 5]
df[df.apply(lambda row: row['A'] + row['B'] > 10, axis=1)]
```

---

## **Chained Filtering**
Multiple filters in one line.

```python
df[(df['A'] > 5) & (df['B'] < 10)]
```

---

## **Custom Function-Based Filtering**
Pass custom logic with `.apply()`.

```python
def is_valid(row):
    return row['score'] > 80 and row['grade'] == 'A'

df[df.apply(is_valid, axis=1)]
```

---

## **Deprecated or Avoided**
- `df.ix[]` ❌ (Deprecated)
- `df.get_value()` ❌ (Use `.at[]` instead)

---

## **Best Practices**
- Use `.query()` for readability in complex logic
- Avoid chained indexing like `df[df['A'] > 5]['B']` (use `.loc[]` instead)
- Use `~` and `isin()` for cleaner negation logic

---
