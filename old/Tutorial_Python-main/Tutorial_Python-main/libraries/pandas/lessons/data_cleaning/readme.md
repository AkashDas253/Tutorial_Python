## Data Cleaning in Pandas

### Purpose

Data cleaning is the process of **preparing raw data for analysis** by correcting errors, handling inconsistencies, filling gaps, and ensuring uniformity. Clean data leads to accurate analysis, better model performance, and improved interpretability.

---

### Key Goals of Data Cleaning

| Goal                     | Focus                                |
| ------------------------ | ------------------------------------ |
| Handle missing values    | Ensure continuity and completeness   |
| Remove duplicates        | Ensure uniqueness and avoid bias     |
| Correct data types       | Prevent type-related errors          |
| Fix inconsistent formats | Standardize values and structure     |
| Detect anomalies         | Identify and flag outliers or errors |
| Rename/relabel           | Improve readability and consistency  |
| Normalize structure      | Flatten or reshape for uniformity    |

---

### Major Cleaning Operations

#### Missing Values Handling

* Detect and quantify missingness
* Options:

  * Fill with default/statistical values
  * Forward/backward fill
  * Interpolate values
  * Drop rows/columns

#### Duplicate Detection

* Identify duplicate rows (partial or full)
* Remove while preserving one occurrence
* Use custom subset of columns to check

#### Type Conversion

* Ensure columns have appropriate dtypes
* Convert object columns to numeric, datetime, category, etc.
* Use inferencing or explicit casting

#### String Cleaning & Normalization

* Strip extra spaces, symbols
* Standardize case, formatting
* Replace unwanted substrings
* Normalize encoding (e.g., Unicode cleanup)

#### Renaming

* Rename columns to consistent naming conventions
* Set index names or reset if needed

#### Value Correction

* Correct known data entry errors
* Replace placeholder or invalid values (e.g., '-999', 'N/A')

#### Column/Row Dropping

* Remove irrelevant, redundant, or empty fields
* Drop by name or based on content (e.g., % missing)

---

### Commonly Used Tools

| Tool                    | Focus                                     |
| ----------------------- | ----------------------------------------- |
| `isnull()`, `notnull()` | Missing data mask                         |
| `fillna()`              | Impute missing values                     |
| `dropna()`              | Remove nulls                              |
| `duplicated()`          | Flag duplicates                           |
| `drop_duplicates()`     | Remove duplicate entries                  |
| `astype()`              | Change data types                         |
| `replace()`             | Replace specific values                   |
| `str` accessor          | String cleaning operations                |
| `rename()`              | Rename columns/index                      |
| `drop()`                | Drop rows or columns                      |
| `interpolate()`         | Estimate missing values based on patterns |

---

### Best Practices

* Always profile data before cleaning
* Work on copies or use `inplace=False` to avoid unintentional overwrite
* Combine logical filters with cleaning steps (e.g., drop rows with outliers + nulls)
* Clean data incrementally and validate each step
* Document each cleaning decision for reproducibility

---

### Integration with Workflow

Data cleaning typically happens:

* **Before EDA**: To ensure summaries reflect true data
* **Before modeling**: To avoid bias and errors
* **After merging**: To resolve inconsistencies across sources
* **During feature engineering**: To prep derived attributes

---
