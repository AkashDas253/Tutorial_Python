## Data Manipulation in Pandas

### Purpose

Data manipulation refers to **modifying, transforming, or restructuring** data to make it suitable for analysis, visualization, or modeling. It includes creating new columns, altering values, reordering data, and combining or reshaping datasets.

---

### Key Objectives of Data Manipulation

| Objective               | Description                             |
| ----------------------- | --------------------------------------- |
| Add or remove columns   | Refine the structure of the dataset     |
| Modify values           | Adjust or derive data for new meaning   |
| Reorder or rename       | Improve readability and processing      |
| Sort or rank data       | Arrange for presentation or logic       |
| Map or transform values | Apply logic or rules across data        |
| Combine datasets        | Merge information from multiple sources |

---

### Common Manipulation Actions

#### Column & Row Modification

* Add new columns from formulas or transformations
* Drop unused or irrelevant columns/rows
* Rename columns for clarity or consistency

#### Sorting

* Sort rows by column values
* Sort indexes for alignment or ordered processing
* Support for ascending/descending and multi-column sorting

#### Value Transformation

* Apply custom or built-in functions to columns
* Use row-wise or element-wise operations
* Normalize, scale, or round values

#### Mapping & Encoding

* Replace values using dictionaries or functions
* Map categorical values to numbers or labels
* Use `.map()`, `.replace()`, or `.apply()`

#### Reindexing & Renaming

* Reset or redefine the index
* Align data to a new index structure
* Rename labels without altering data

#### Apply Functions

* Apply functions column-wise or row-wise
* Use `.apply()` for flexible logic
* Use `.applymap()` for element-wise ops in DataFrames
* Use `.pipe()` for chainable transformations

---

### Tools and Operations

| Operation               | Description                              |
| ----------------------- | ---------------------------------------- |
| `assign()`              | Add new columns with chainable syntax    |
| `drop()`                | Remove rows or columns                   |
| `rename()`              | Change names of columns or index labels  |
| `sort_values()`         | Sort data by column values               |
| `sort_index()`          | Sort data by index                       |
| `apply()`               | Apply function across rows/columns       |
| `map()`                 | Element-wise mapping (Series only)       |
| `replace()`             | Replace values conditionally             |
| `applymap()`            | Element-wise operations across DataFrame |
| `round()`               | Round numerical data                     |
| `clip()`                | Limit values within a range              |
| `rank()`                | Assign ranking within columns            |
| `cumsum()`, `cumprod()` | Cumulative operations for trend analysis |
| `pipe()`                | Enable functional chaining of operations |

---

### Strategic Use Cases

| Use Case                   | Description                                                 |
| -------------------------- | ----------------------------------------------------------- |
| Feature engineering        | Derive new features from existing columns                   |
| Preprocessing for modeling | Normalize, encode, scale values                             |
| Data standardization       | Format or transform entries for consistency                 |
| Data labeling              | Attach readable labels to numeric codes                     |
| Conditional logic          | Apply custom rules across data                              |
| Aggregated transformation  | Transform group-level values using groupby with `transform` |

---

### Workflow Integration

Data manipulation typically occurs:

* **Post data cleaning**: To derive or restructure for modeling
* **Before analysis**: To shape data for specific visual or statistical tools
* **After aggregation**: To format or expand results for further use
* **During feature selection**: To drop or transform redundant variables

---
