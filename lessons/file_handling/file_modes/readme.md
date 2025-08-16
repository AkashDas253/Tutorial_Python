# File Modes in Python 

File modes determine **how a file is opened** (read/write/append) and **in what format** (text/binary). They control the behavior of file pointers, whether the file gets truncated, and whether a new file is created.

---

## Classification of File Modes

### Text Modes

* `"r"` → Read (default)

  * Opens for reading only.
  * File must exist.
  * Pointer at beginning.

* `"w"` → Write

  * Opens for writing only.
  * Truncates (clears) file if it exists.
  * Creates new file if not exists.

* `"a"` → Append

  * Opens for writing only.
  * Creates file if not exists.
  * Pointer always at end (content preserved).

* `"r+"` → Read & Write

  * File must exist.
  * Pointer at beginning.
  * Can overwrite existing content.

* `"w+"` → Write & Read

  * Truncates file if exists.
  * Creates new file if not exists.
  * Pointer at beginning.

* `"a+"` → Append & Read

  * Creates file if not exists.
  * Pointer at end for writing, beginning for reading after `seek(0)`.

---

### Binary Modes

Add `"b"` to any mode for **binary operations**.

* `"rb"` → Read binary file.
* `"wb"` → Write binary file (truncate/overwrite).
* `"ab"` → Append binary data.
* `"rb+"` → Read & write binary (no truncation, pointer at beginning).
* `"wb+"` → Write & read binary (truncate/overwrite).
* `"ab+"` → Append & read binary.

---

## Syntax

```python
file = open("filename.txt", mode="r", encoding="utf-8")  # text mode
file = open("filename.bin", mode="rb")  # binary mode
```

---

## File Mode Behavior Comparison

| Mode | File must exist? | Truncate? | Create if not exist? | Pointer Position |
| ---- | ---------------- | --------- | -------------------- | ---------------- |
| r    | ✅ Yes            | ❌ No      | ❌ No                 | Start            |
| w    | ❌ No             | ✅ Yes     | ✅ Yes                | Start            |
| a    | ❌ No             | ❌ No      | ✅ Yes                | End              |
| r+   | ✅ Yes            | ❌ No      | ❌ No                 | Start            |
| w+   | ❌ No             | ✅ Yes     | ✅ Yes                | Start            |
| a+   | ❌ No             | ❌ No      | ✅ Yes                | End (write)      |

---

## Usage Scenarios

* **r** → Reading configurations, data files.
* **w** → Writing reports, overwriting logs.
* **a** → Logging, incremental data collection.
* **r+** → Updating part of an existing file.
* **w+** → Resetting and writing fresh data with future reads.
* **a+** → Logs where you may also need to read previous entries.
* **Binary modes** → Images, audio, pickle files.

---
