## Sessions

### About Sessions

- In TensorFlow 1.x, **sessions** were used to execute operations in the computational graph and manage resources. 
- However, in TensorFlow 2.x, eager execution is enabled by default, which means that operations run immediately as they are called, without the need for an explicit session. 
- If you are still working with TensorFlow 1.x or using the low-level TensorFlow 2.x API in graph mode, you may need to understand how sessions work.

---

### Creating and Running a TensorFlow Session

- A **TensorFlow session** (`tf.compat.v1.Session` in TensorFlow 2.x) is an environment that manages the execution of the computational graph, allowing variables to be initialized, tensors to be evaluated, and operations to be run.



### 1. **Creating a TensorFlow Session**

- The primary function for creating a session in TensorFlow 1.x (or 2.x compatibility mode) is `tf.compat.v1.Session`.

- Syntax:
    ```python
    session = tf.compat.v1.Session(target='', graph=None, config=None)
    ```

- **Parameters**:
  - **`target`** (default: `''`): A string specifying the execution engine (if distributed execution is used). Usually left as an empty string for local execution.
  - **`graph`** (default: `None`): The `tf.Graph` to be launched in this session. If `None`, the default graph is used.
  - **`config`** (default: `None`): A `tf.compat.v1.ConfigProto` protocol buffer to specify configuration options for the session (such as device placements and GPU options).

Example:
```python
import tensorflow as tf

# Start a session
session = tf.compat.v1.Session()
```

### 2. **Using `ConfigProto` for Session Configuration**

- `tf.compat.v1.ConfigProto` allows you to customize the session with various configuration options, which can be helpful for performance tuning, memory allocation, and device placement.

- `tf.compat.v1.ConfigProto`:

- Syntax:
    ```python
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    ```

- **Parameters**:
  - **`allow_soft_placement`** (default: `False`): Automatically places operations on available devices if the specified one is unavailable.
  - **`log_device_placement`** (default: `False`): Logs device placement for debugging purposes.
  - **`gpu_options`**: Contains settings for GPU management, such as limiting GPU memory usage.
  
Example:
```python
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
session = tf.compat.v1.Session(config=config)
```

### 3. **Running Operations in a Session**

- Once a session is created, you can execute operations using the `.run()` method. 
- This is where you specify which operations (like adding, multiplying) to execute and optionally pass in any required feed data.

- Syntax:
    ```python
    session.run(fetches, feed_dict=None, options=None, run_metadata=None)
    ```

- **Parameters**:
  - **`fetches`**: The operation or list of operations to execute (e.g., tensors or `tf.Operation` objects).
  - **`feed_dict`** (optional, default: `None`): A dictionary that allows you to override the values of tensors within the graph. Used for passing input values.
  - **`options`** (optional, default: `None`): A `tf.compat.v1.RunOptions` object that provides options for tracing and debugging.
  - **`run_metadata`** (optional, default: `None`): A `tf.compat.v1.RunMetadata` object for collecting metadata about the session run (useful for profiling and analysis).

Example:
```python
# Define operations
a = tf.constant(5.0)
b = tf.constant(3.0)
c = a + b

# Run the operation within a session
result = session.run(c)
print("Result:", result)  # Output: 8.0
```

In this example:
- **`fetches`** is `c`, which represents the addition operation (`a + b`).
- **`feed_dict`** could be used if `a` and `b` were placeholders that need external values.

### 4. **Using `feed_dict` for Input Data**

The `feed_dict` parameter in `session.run()` allows you to pass data into the graph dynamically. This is particularly useful when working with placeholders.

Example:
```python
# Define placeholders
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
add_op = x + y

# Execute with values for placeholders
result = session.run(add_op, feed_dict={x: 4.0, y: 2.0})
print("Feed dict result:", result)  # Output: 6.0
```

### 5. **Closing a Session**

- It’s essential to free up resources by closing the session once it is no longer needed.

    ```python
    session.close()
    ```

- Alternatively, you can use a session within a `with` block, which will automatically close it at the end:

    ```python
    with tf.compat.v1.Session() as session:
        result = session.run(add_op, feed_dict={x: 5.0, y: 3.0})
        print("With block result:", result)  # Output: 8.0
    ```

---

### Summary Table

| Function/Parameter       | Description                                                                                 | Default Value           |
|--------------------------|---------------------------------------------------------------------------------------------|--------------------------|
| `tf.compat.v1.Session`   | Creates a TensorFlow session for graph execution.                                           | -                        |
| `target`                 | Specifies execution target (e.g., for distributed execution).                               | `''` (local execution)   |
| `graph`                  | Specifies the computational graph to launch.                                                | `None` (default graph)   |
| `config`                 | Configuration options for session, specified as `ConfigProto`.                             | `None`                   |
| `allow_soft_placement`   | Allows operations to be placed on alternative devices if required device is unavailable.    | `False`                  |
| `log_device_placement`   | Logs the device placement of operations for debugging.                                      | `False`                  |
| `session.run()`          | Executes operations or tensors within the graph.                                            | -                        |
| `fetches`                | The operations/tensors to execute.                                                          | -                        |
| `feed_dict`              | Dictionary for feeding values to placeholders.                                              | `None`                   |
| `options`                | RunOptions for tracing and debugging.                                                       | `None`                   |
| `run_metadata`           | RunMetadata for collecting metadata on session run.                                         | `None`                   |

---

### Example: Complete TensorFlow 1.x Session Workflow

```python
import tensorflow as tf

# Define constants and operations
a = tf.constant(5.0)
b = tf.constant(3.0)
c = a * b

# Create session configuration
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# Run the session
with tf.compat.v1.Session(config=config) as session:
    result = session.run(c)
    print("Result from session:", result)  # Output: 15.0
```

This code demonstrates the complete workflow of defining constants, configuring the session, and running operations.

---

### Key Points

- **Sessions** manage resources and execute graphs, especially important in TensorFlow 1.x.
- **`tf.compat.v1.Session()`** creates a session in TensorFlow 2.x, allowing access to TensorFlow 1.x features.
- **`feed_dict`** provides a way to dynamically pass input data.
- **`ConfigProto`** optimizes session configuration, offering options like soft placement and logging device placements.

This basic understanding of TensorFlow sessions is essential for using TensorFlow 1.x and for leveraging low-level graph execution in TensorFlow 2.x compatibility mode.
