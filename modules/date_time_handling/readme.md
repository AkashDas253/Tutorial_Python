# Date & Time Handling in Python 

## Core Concepts

* **Date** → Represents calendar date (year, month, day).
* **Time** → Represents clock time (hours, minutes, seconds, microseconds).
* **DateTime** → Combines date + time into one object.
* **Timestamp** → Numeric representation of time (usually seconds since Unix epoch `1970-01-01 00:00:00 UTC`).
* **Timezone** → Geographic offset rules (UTC vs local, daylight savings).
* **Intervals** → Differences or durations between times (e.g., `timedelta`).
* **Formatting/Parsing** → Converting between `datetime` objects and string representations.
* **Calendrical Systems** → Handling weeks, weekdays, leap years, holidays.
* **Clocks** → System-based time readings (`wall-clock`, `monotonic`, `perf_counter`).

---

## Python Standard Library Tools

### `datetime`

* Primary tool for representing **dates, times, and combined datetimes**.
* Provides classes: `date`, `time`, `datetime`, `timedelta`, `timezone`.
* Used for:

  * Creating/manipulating dates and times.
  * Time arithmetic (`now + timedelta(days=5)`).
  * Formatting/parsing with `strftime` & `strptime`.
  * Timezone handling with `timezone` (basic) and `zoneinfo` (detailed).

### `time`

* Direct interface to **POSIX time functions**.
* Used for:

  * Reading current epoch (`time.time()`).
  * Pausing (`time.sleep()`).
  * Measuring intervals (`time.perf_counter()`, `time.monotonic()`).
  * Conversion helpers (`gmtime`, `localtime`, `strftime`).

### `calendar`

* Provides **calendar-based utilities**.
* Used for:

  * Generating month/year calendars.
  * Checking leap years.
  * Getting weekdays and week structures.
  * Useful for human-readable or scheduling logic.

### `zoneinfo` (Python 3.9+)

* Official IANA **time zone support**.
* Handles regional rules like daylight savings.
* Integrated with `datetime`.

---

## Extended Libraries (for advanced workflows)

* **`dateutil`** → Parsing natural language dates, recurrence rules.
* **`pytz`** → Legacy timezone library (superseded by `zoneinfo`).
* **`pendulum`** → User-friendly timezone-aware replacement for `datetime`.
* **`arrow`** → Simplified date/time handling, humanized intervals ("3 hours ago").

---

## Common Usage Scenarios

### Representing and Calculating Time

* Current moment (`datetime.now()`, `time.time()`).
* Future/past (`datetime.now() + timedelta(days=7)`).
* Duration calculation (`end - start`).

### Formatting & Parsing

* Convert to string (`strftime("%Y-%m-%d %H:%M")`).
* Parse string to datetime (`strptime("2025-08-16", "%Y-%m-%d")`).

### Time Zones

* Convert naive datetime → aware datetime with `zoneinfo.ZoneInfo("Asia/Kolkata")`.
* Shift between UTC/local.
* Handle daylight savings changes safely.

### Scheduling / Waiting

* `time.sleep()` for delays.
* `sched` or external schedulers for jobs.
* Monotonic/perf counters for benchmarking.

### Calendar Operations

* Generate month/year views.
* Find weekdays, leap years, holidays.
* Scheduling recurring tasks with `dateutil.rrule`.

---

## Conceptual Workflow

1. **Choose representation** → `datetime` vs `timestamp`.
2. **Attach timezone** if dealing with global data.
3. **Perform arithmetic** with `timedelta`.
4. **Format/parse** for input-output.
5. **Use monotonic clocks** for reliability in measuring durations.
6. **Use calendar/timezone utilities** for recurring or regional rules.

---
