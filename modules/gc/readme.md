# `gc` Module – Garbage Collection

The `gc` module provides an interface to Python’s garbage collector for controlling, debugging, and tuning memory management of cyclic references.

---

## Core Concepts

* **Reference counting**: Python uses reference counting as the primary memory management.
* **Garbage collector**: Supplements ref counting by detecting & collecting **cyclic references** (objects referencing each other).
* **Generational GC**: Objects are divided into generations; younger generations are collected more often.

---

## Functions

### Collection Control

* `gc.collect(generation=2)`
  Force garbage collection.

  * `generation`: 0 (young), 1 (intermediate), 2 (all, default).
  * Returns number of unreachable objects found.

* `gc.enable()`
  Enable automatic garbage collection.

* `gc.disable()`
  Disable automatic garbage collection.

* `gc.isenabled()`
  Check if automatic collection is enabled.

---

### Threshold & Statistics

* `gc.set_threshold(threshold0, threshold1, threshold2)`
  Configure collection frequency.

  * `threshold0`: # of allocations before collecting generation 0.
  * `threshold1`: Generation 1 threshold.
  * `threshold2`: Generation 2 threshold.

* `gc.get_threshold()`
  Get current thresholds.

* `gc.get_count()`
  Return number of allocations since last collection (per generation).

* `gc.get_stats()`
  Return stats per generation (since Python 3.4+).

---

### Debugging

* `gc.set_debug(flags)`
  Enable debugging output. Flags:

  * `gc.DEBUG_STATS` – Print collection stats
  * `gc.DEBUG_COLLECTABLE` – Print collectable objects
  * `gc.DEBUG_UNCOLLECTABLE` – Print uncollectable objects
  * `gc.DEBUG_SAVEALL` – Save all unreachable objects in `gc.garbage`

* `gc.get_debug()`
  Return current debug flags.

* `gc.garbage`
  List of uncollectable objects (with `DEBUG_SAVEALL`).

---

### Object Access

* `gc.get_objects()`
  Return list of all tracked objects.

* `gc.get_referrers(*objs)`
  Return objects that refer to given objects.

* `gc.get_referents(*objs)`
  Return objects referred to by given objects.

* `gc.is_tracked(obj)`
  Check if object is tracked by GC.

* `gc.freeze()`
  Move all objects to permanent generation (avoid collection).

* `gc.unfreeze()`
  Re-enable collection for frozen objects.

---

## Usage Example

```python
import gc

# Disable automatic GC
gc.disable()
print("GC enabled:", gc.isenabled())  # False

# Manually collect garbage
unreachable = gc.collect()
print("Unreachable objects collected:", unreachable)

# Set thresholds
gc.set_threshold(700, 10, 10)
print("Thresholds:", gc.get_threshold())

# Debugging
gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_COLLECTABLE)

# Accessing objects
obj_list = []
print("Is obj_list tracked?", gc.is_tracked(obj_list))

# Forcing collection of all generations
gc.collect(2)
```

---

## When to Use

* Debugging memory leaks.
* Forcing collection in memory-sensitive apps.
* Profiling memory-heavy code.
* Disabling/enabling GC for performance tuning (e.g., during batch allocations).

---
