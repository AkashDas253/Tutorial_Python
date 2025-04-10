
## **Date and Time Handling in Pandas**

Pandas provides extensive support for **date and time manipulation**, crucial for time series analysis, indexing, filtering, and feature engineering.

---

###  **Date and Time Data Types**

| Type                    | Description                                 | Example                     |
|-------------------------|---------------------------------------------|-----------------------------|
| `datetime64[ns]`        | Timestamp (date + time, nanosecond-level)   | `2023-04-01 12:00:00`       |
| `timedelta64[ns]`       | Difference between two datetime values      | `5 days 02:00:00`           |
| `Period`                | Time span (e.g., month, quarter)            | `2024-12` (Monthly Period)  |
| `Timestamp`             | Single point in time (wrapper of datetime)  | `pd.Timestamp('2024-01-01')`|
| `DatetimeIndex`         | Index of timestamps                         | Used in time series         |
| `TimedeltaIndex`        | Index of time deltas                        | Used for intervals          |
| `PeriodIndex`           | Index for periods                           | `pd.period_range(...)`      |

---

###  **Creating Date/Time Objects**

```python
pd.to_datetime(arg, format=None, errors='raise')
```

| Parameter   | Description                                                  |
|-------------|--------------------------------------------------------------|
| `arg`       | String, list, array, or Series of dates                      |
| `format`    | Custom date-time format (`%Y-%m-%d`, etc.)                   |
| `errors`    | Options: `'raise'`, `'coerce'`, `'ignore'`                   |

**Examples:**

```python
pd.to_datetime('2024-01-01')                          # Single timestamp  
pd.to_datetime(['2024-01-01', '2024-01-05'])          # Array of timestamps  
pd.date_range(start='2024-01-01', periods=5, freq='D')# Date range
```

---

###  **Date Range Generators**

```python
pd.date_range(start=None, end=None, periods=None, freq='D')
```

| Parameter   | Description                                 |
|-------------|---------------------------------------------|
| `start/end` | Start and end dates                         |
| `periods`   | Number of periods to generate               |
| `freq`      | Frequency: `'D'`, `'M'`, `'H'`, `'Y'`, etc. |

**Examples:**

```python
pd.date_range('2024-01-01', periods=5, freq='D')       # Daily  
pd.date_range('2024-01-01', '2024-01-10', freq='2D')   # Every 2 days  
pd.date_range('2024-01-01', periods=5, freq='M')       # Monthly
```

---

###  **Datetime Properties**

When a column is of `datetime64[ns]` type, `.dt` accessor exposes many datetime properties:

| Property        | Description               | Example                     |
|----------------|---------------------------|-----------------------------|
| `.dt.year`      | Year                      | `df['date'].dt.year`        |
| `.dt.month`     | Month                     | `df['date'].dt.month`       |
| `.dt.day`       | Day of month              | `df['date'].dt.day`         |
| `.dt.hour`      | Hour                      | `df['date'].dt.hour`        |
| `.dt.minute`    | Minute                    | `df['date'].dt.minute`      |
| `.dt.second`    | Second                    | `df['date'].dt.second`      |
| `.dt.dayofweek` | 0 = Monday … 6 = Sunday   | `df['date'].dt.dayofweek`   |
| `.dt.date`      | Return date only          | `df['date'].dt.date`        |
| `.dt.time`      | Return time only          | `df['date'].dt.time`        |
| `.dt.week` / `.dt.isocalendar().week` | Week number | `df['date'].dt.isocalendar().week` |
| `.dt.quarter`   | Quarter of year (1–4)     | `df['date'].dt.quarter`     |
| `.dt.daysinmonth` | Days in month           | `df['date'].dt.daysinmonth` |
| `.dt.is_month_end` | Boolean if month end   | `df['date'].dt.is_month_end`|

---

###  **Timedeltas (Date Differences)**

```python
df['delta'] = df['end_date'] - df['start_date']  
# Results in timedelta64
```

**Timedelta Properties**

| Property         | Description                       |
|------------------|-----------------------------------|
| `.days`          | Number of full days               |
| `.seconds`       | Remaining seconds                 |
| `.total_seconds()` | Total seconds (including days) |

---

###  **Period and PeriodIndex**

```python
pd.Period('2024-12', freq='M')  
# Single Period (Monthly)
```

```python
pd.period_range('2024-01', '2024-06', freq='M')  
# PeriodIndex
```

```python
df['period'] = pd.to_datetime(df['date']).dt.to_period('M')  
# Convert datetime to Period
```

---

###  **Conversion Between Types**

```python
pd.to_datetime()        # String → datetime  
pd.to_timedelta()       # String/num → timedelta  
datetime.to_period()    # Datetime → Period  
period.to_timestamp()   # Period → Datetime  
```

---

###  **Date Filtering Examples**

```python
df[df['date'] >= '2024-01-01']  
df[df['date'].dt.month == 1]  
df[df['date'].between('2024-01-01', '2024-02-01')]
```

---

###  **Resampling by Time**

```python
df.resample('M').mean()      # Monthly average  
df.resample('W').sum()       # Weekly sum  
df.resample('Q').first()     # Quarterly first value
```

> Requires a datetime-like index.

---

