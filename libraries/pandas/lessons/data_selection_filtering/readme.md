## Selection & Filtering in Pandas

### Purpose

Selection and filtering are core to narrowing down data for analysis by **extracting specific rows, columns, or values** based on labels, positions, or conditions. It enables focused inspection, modification, or computation on subsets of data.

---

### Key Concepts

| Concept             | Description                                     |
| ------------------- | ----------------------------------------------- |
| **Label-based**     | Accessing data using explicit row/column labels |
| **Position-based**  | Accessing data by numerical indices             |
| **Boolean masking** | Filtering based on conditional logic            |
| **Callable logic**  | Passing functions to dynamically select/filter  |

---

### Types of Selection

####  Column Selection

* Selecting one or more columns from a DataFrame
* Used for narrowing features or inspecting single dimensions

####  Row Selection

* Based on label or index (named rows)
* Based on integer position (row number)
* Useful for retrieving specific observations

####  Combined Row-Column Selection

* Simultaneous control of rows and columns
* Enables subsetting by conditions and selecting only relevant fields

---

### Selection Tools

| Tool       | Purpose                                   |
| ---------- | ----------------------------------------- |
| `[]`       | Quick column access or row slicing        |
| `.loc[]`   | Label-based selection of rows/columns     |
| `.iloc[]`  | Integer-based selection                   |
| `.at[]`    | Fast access to single value by label      |
| `.iat[]`   | Fast access to single value by index      |
| `filter()` | Subset by column/row names using patterns |

---

### Filtering Techniques

| Technique                | Focus                                       |          |
| ------------------------ | ------------------------------------------- | -------- |
| Boolean conditions       | Select rows where condition is True         |          |
| Multiple conditions      | Combine filters using `&`, \`               | `, `\~\` |
| Value membership         | Select rows where column value is in a list |          |
| Range-based filtering    | Select values between a min and max         |          |
| String-based filtering   | Filter using `.str` methods                 |          |
| Function-based filtering | Dynamic row filtering using custom logic    |          |

---

### Usage Insights

* Always distinguish between **label-based** (`loc`) and **position-based** (`iloc`)
* When combining conditions, wrap each condition in parentheses
* For column filtering based on patterns, `filter()` with `like=`, `regex=`, or `items=` is useful
* `query()` provides SQL-like syntax for filtering
* Boolean masks can be reused to apply across different operations

---

### Common Patterns

| Pattern                       | Description                             |
| ----------------------------- | --------------------------------------- |
| Select columns dynamically    | Based on name pattern or metadata       |
| Filter rows by condition      | Logical operations on columns           |
| Subset columns post-filtering | Efficient chainable filtering           |
| Mask + assign                 | Modify values where condition holds     |
| Drop based on condition       | Remove rows or columns that match logic |

---

### Integration

Selection and filtering are often used with:

* **GroupBy**: Select and then aggregate
* **Apply**: Run custom logic after filtering
* **Merge**: Join only relevant slices
* **Plotting**: Visualize filtered subsets
* **Cleaning**: Remove outliers, nulls, or redundant values

---
