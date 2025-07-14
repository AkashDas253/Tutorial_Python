# System related 


## File Handling

- File handling is an important part of any web application.
- Python has several functions for creating, reading, updating, and deleting files.

### File Opening and closing

#### Open file:

```python
var = open(filename, mode="rt")
```

- `open(filename, mode="r")`: Opens a file and returns a file object.
  - `filename`: The name of the file to be opened.
  - `mode`: The mode in which the file is opened. Default is `"r"` (read mode).
    - There are four different methods (modes) for opening a file:
        - `"r"` - Read - Default value. Opens a file for reading, error if the file does not exist
        - `"a"` - Append - Opens a file for appending, creates the file if it does not exist
        - `"w"` - Write - Opens a file for writing, creates the file if it does not exist
        - `"x"` - Create - Creates the specified file, returns an error if the file exists

    - In addition, you can specify if the file should be handled as binary or text mode:
        - `"t"` - Text - Default value. Text mode
        - `"b"` - Binary - Binary mode (e.g. images)


#### Closing:

```python
var.close()
```

- `close()`: Closes the file. No parameters.

### With Open (Safe Usage)

#### Syntax:

```python
with open(filename, mode="r") as file_object:
    # Perform file operations
    # No need to explicitly close the file; it is automatically closed when the block is exited.
```

- `filename`: The name of the file to be opened.
- `mode`: The mode in which the file is opened. Default is `"r"` (read mode).
- `file_object`: The file object that you can use to perform file operations.

#### Benefits

- **Automatic Resource Management:** Ensures that the file is properly closed after its suite finishes.
- **Cleaner Code:** Reduces the need for explicit `try...finally` blocks to close the file.
- **Exception Safety:** Even if an exception occurs within the block, the file will still be closed properly.

### File Write and Read:

- `read(size=-1)`: Reads the entire content of the file or up to `size` bytes.
  - `size`: The number of bytes to read. Default is `-1` (read all).

- `readline(size=-1)`: Reads a single line from the file or up to `size` bytes.
  - `size`: The number of bytes to read. Default is `-1` (read the entire line).

- `readlines(hint=-1)`: Reads all the lines of a file and returns them as a list.
  - `hint`: The number of lines to read. Default is `-1` (read all lines).

- `write(string)`: Writes a string to the file.
  - `string`: The string to be written to the file.

- `writelines(lines)`: Writes a list of strings to the file.
  - `lines`: A list of strings to be written to the file.

- `close()`: Closes the file. No parameters.

- `flush()`: Flushes the internal buffer. No parameters.

### File Object Methods

- `fileno()`: Returns the file descriptor. No parameters.

- `isatty()`: Returns `True` if the file is connected to a terminal device. No parameters.

- `readable()`: Returns `True` if the file can be read. No parameters.

- `writable()`: Returns `True` if the file can be written to. No parameters.

- `seekable()`: Returns `True` if the file supports random access. No parameters.