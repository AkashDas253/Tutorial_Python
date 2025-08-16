# `datetime` Module in Python

The `datetime` module in Python provides classes for working with dates and times, including parsing, formatting, arithmetic, and timezone handling.

---

## Components

### `date` (Represents a calendar date)

* Attributes: `.year`, `.month`, `.day`
* Methods:

  * `today()`
  * `fromtimestamp(timestamp)`
  * `isoformat()`
  * `weekday()`, `isoweekday()`

### `time` (Represents a time of day)

* Attributes: `.hour`, `.minute`, `.second`, `.microsecond`, `.tzinfo`
* Methods:

  * `isoformat()`
  * `replace(hour, minute, ...)`

### `datetime` (Combines date & time)

* Attributes: `.year`, `.month`, `.day`, `.hour`, `.minute`, `.second`, `.microsecond`, `.tzinfo`
* Methods:

  * `now(tz=None)`
  * `utcnow()`
  * `combine(date, time)`
  * `fromtimestamp(timestamp)`
  * `strptime(date_string, format)`
  * `strftime(format)`
  * Arithmetic with `timedelta`

### `timedelta` (Represents a duration)

* Attributes: `.days`, `.seconds`, `.microseconds`
* Supports arithmetic (`+`, `-`) with `date`/`datetime`
* Useful for date calculations

### `tzinfo` & `timezone` (Timezone support)

* `tzinfo`: Abstract base for time zones
* `timezone`: Fixed offset timezone (`timezone.utc`, `timezone(timedelta(hours=5.5))`)

---

## Formatting & Parsing

* `strftime(format)` → Format datetime object into string
* `strptime(date_string, format)` → Parse string into datetime

Common format codes:

* `%Y` → Year, `%m` → Month, `%d` → Day
* `%H` → Hour, `%M` → Minute, `%S` → Second
* `%A` → Weekday name, `%B` → Month name

---

## Examples

```python
from datetime import datetime, date, time, timedelta, timezone

# Current date and time
now = datetime.now()
print("Now:", now)

# Create specific date
d = date(2025, 8, 16)
print("Date:", d, "Weekday:", d.weekday())

# Create specific time
t = time(14, 30, 45)
print("Time:", t)

# Datetime from timestamp
dt = datetime.fromtimestamp(1692192000)
print("From timestamp:", dt)

# String formatting
formatted = now.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted:", formatted)

# String parsing
parsed = datetime.strptime("2025-08-16 14:30:00", "%Y-%m-%d %H:%M:%S")
print("Parsed:", parsed)

# Date arithmetic
tomorrow = now + timedelta(days=1)
print("Tomorrow:", tomorrow)

# Timezone aware datetime
utc_time = datetime.now(timezone.utc)
print("UTC Time:", utc_time)
```

---
