# `calendar` Module in Python

The `calendar` module provides functions and classes to work with dates, calendars, and related formatting. It supports generating plain text, HTML calendars, and checking properties of dates.

---

## Key Features

* Calendar generation (text and HTML)
* Locale support for month/day names
* Date calculations (leap years, weekdays)
* Iterators for weeks/months

---

## Classes

* **`Calendar`** – Base class for iterating over weeks/months.
* **`TextCalendar`** – Generates calendars in plain text.
* **`HTMLCalendar`** – Generates calendars in HTML format.
* **`LocaleTextCalendar`** – Locale-aware text calendar.
* **`LocaleHTMLCalendar`** – Locale-aware HTML calendar.

---

## Functions

### General Calendar Operations

* `calendar(year, w=2, l=1, c=6, m=3)` → multi-column year calendar as string
* `prcal(year, w=2, l=1, c=6, m=3)` → print year calendar
* `month(year, month, w=2, l=1)` → month as string
* `prmonth(year, month, w=2, l=1)` → print month calendar

### Date Checks

* `isleap(year)` → check leap year (bool)
* `leapdays(y1, y2)` → number of leap years in range `[y1, y2)`
* `weekday(year, month, day)` → return day of week (`0=Monday`)

### Iterators

* `monthcalendar(year, month)` → matrix of weeks (lists of ints, `0=empty`)
* `monthrange(year, month)` → `(weekday, days_in_month)`
* `yeardayscalendar(year, width=3)` → list of months grouped into weeks
* `yeardays2calendar(year, width=3)` → like above with day numbers and weekdays

### Localization

* `month_name` → list of month names (1–12, index 0 empty)
* `month_abbr` → list of abbreviated month names
* `day_name` → list of weekday names (0=Monday)
* `day_abbr` → list of weekday abbreviations

---

## Examples

```python
import calendar

# Print a month
print(calendar.month(2025, 8))

# Check leap year
print(calendar.isleap(2024))  # True

# Get weekday of a date
print(calendar.weekday(2025, 8, 16))  # 5 (Saturday)

# Month as a matrix
print(calendar.monthcalendar(2025, 8))

# Using TextCalendar
tc = calendar.TextCalendar(firstweekday=6)  # Sunday start
print(tc.formatmonth(2025, 8))

# Using HTMLCalendar
hc = calendar.HTMLCalendar(firstweekday=0)
print(hc.formatmonth(2025, 8))
```

---
