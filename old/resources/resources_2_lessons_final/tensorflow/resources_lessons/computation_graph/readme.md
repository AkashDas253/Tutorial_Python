## Computational Graph

- In TensorFlow, a **computational graph** is a directed graph where nodes represent operations (like addition or multiplication), and edges represent the data (tensors) that flow between these operations. 
- The computational graph serves as the foundation for efficient parallel computation, optimization, and gradient-based training in TensorFlow.

### **About Computational Graph**

- A computational graph is a structured way to represent mathematical operations, where:
    - **Nodes** represent operations.
    - **Edges** represent tensors that flow between operations.
  
- TensorFlow 2.x uses **eager execution** by default, which evaluates operations immediately. 
- However, computational graphs are still crucial for performance optimization, especially in high-performance training scenarios. 
- TensorFlow 2.x introduces the **`tf.function`** decorator, allowing you to create graphs while maintaining an eager execution interface.

---

### 1. **Defining Computational Graphs with `tf.function`**

- In TensorFlow 2.x, you can define a computational graph using the `tf.function` decorator, which converts a Python function into a computation graph. 
- When `tf.function` is applied, TensorFlow traces the function’s operations and constructs a graph.

- `tf.function(func=None, input_signature=None, autograph=True, jit_compile=False)`

- **Parameters**:
  - **`func`**: The function to be converted to a graph. This function should contain TensorFlow operations.
  - **`input_signature`** (optional, default=`None`): A list of `tf.TensorSpec` objects specifying the input shapes and types. Helps with performance and input validation.
  - **`autograph`** (optional, default=`True`): If set to `True`, Python control flow statements like `if`, `while`, and `for` will be converted into TensorFlow operations.
  - **`jit_compile`** (optional, default=`False`): Enables XLA (Accelerated Linear Algebra) compilation for additional optimizations.

**Example**:
```python
import tensorflow as tf

@tf.function
def compute(x, y):
    return x * y + tf.constant(5.0)

# Invoking the function
result = compute(tf.constant(3.0), tf.constant(4.0))
print("Graph Output:", result.numpy())  # Output: 17.0
```

**Notes**:
- **`tf.function`** is used to improve performance by creating a graph for reusable functions.
- **`input_signature`** is useful for defining consistent input shapes and data types, improving performance.


### 2. **Graph Mode and Eager Execution**

- TensorFlow 2.x introduced eager execution, which allows operations to execute immediately, producing concrete values without building graphs. 
- However, using graphs is more efficient for complex model training.

- **Eager Execution**: Operations are evaluated immediately.
  
    ```python
    x = tf.constant(2.0)
    y = x + 3  # Executes immediately
    print(y.numpy())  # Output: 5.0
    ```

- **Graph Mode** (via `tf.function`): Operations are built into a reusable graph, optimizing performance.
    ```python
    @tf.function
    def add(x):
        return x + 3

    # Runs in graph mode, optimized for performance
    output = add(tf.constant(2.0))
    print(output.numpy())  # Output: 5.0
    ```

**Notes**:
- Graphs optimize performance by reducing redundant computations and enabling parallelism.
- Eager execution is convenient for debugging since it produces outputs immediately.

### 3. **Graph Collections and Namespaces**

In TensorFlow, **collections** are used to group related elements, and **namespaces** provide hierarchical organization within the graph.

- **Graph Collections**: Group related operations, variables, or tensors, such as for tracking trainable variables.
```python
tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
```

- **Namespaces**: Help organize and name elements in the graph.
```python
with tf.name_scope("Layer1"):
w = tf.Variable(tf.random.normal([2, 2]), name="weights")
b = tf.Variable(tf.zeros([2]), name="biases")
```

**Notes**:
- **`GraphKeys`** is a collection for commonly used groups like `GLOBAL_VARIABLES` and `TRAINABLE_VARIABLES`.
- **Namespaces** provide logical grouping, making it easier to visualize graphs and locate components.

### 4. **Viewing Computational Graphs with TensorBoard**

TensorBoard, TensorFlow’s visualization toolkit, helps you view the structure of the computational graph.

To use TensorBoard:
1. Write logs to a directory.
   ```python
   writer = tf.summary.create_file_writer("logs")
   ```
2. Use `tf.function` to create a graph and log it.
   ```python
   @tf.function
   def compute(x, y):
       return x + y

   with writer.as_default():
       tf.summary.trace_on(graph=True, profiler=True)
       result = compute(tf.constant(5.0), tf.constant(3.0))
       tf.summary.trace_export(name="compute_graph", step=0, profiler_outdir="logs")
   ```

3. Start TensorBoard by running:
   ```
   tensorboard --logdir=logs
   ```

---

### Example of a Full Computational Graph in TensorFlow

Here’s an example that demonstrates defining a simple graph with constants, variables, and `tf.function`:

```python
import tensorflow as tf

# Define constants
a = tf.constant(2.0, name="constant_a")
b = tf.constant(3.0, name="constant_b")

# Define variable
w = tf.Variable(5.0, name="weight")

# Define a simple computation within a function
@tf.function
def simple_graph(x, y):
    return w * (x + y)

# Run computation
result = simple_graph(a, b)
print("Computed Result:", result.numpy())  # Output: 25.0
```

In this example:
- The constants `a` and `b` represent fixed values.
- The variable `w` is a trainable parameter that could be updated during training.
- The function `simple_graph` builds the graph for the equation \( w \times (a + b) \), optimizing it for performance.

---

### Summary

| Component                  | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| **`tf.function`**          | Converts a Python function into a reusable graph, optimizing it.|
| **Eager Execution**        | Immediate evaluation of operations for interactive programming. |
| **Graph Collections**      | Grouping related operations and variables within the graph.     |
| **Namespaces**             | Logical grouping for clarity in the graph hierarchy.            |
| **TensorBoard**            | Visualization tool for viewing and analyzing computational graphs.|

In TensorFlow 2.x, while eager execution is the default, **graph-based computation** is still key for performance and parallelization, making `tf.function` essential for building efficient models.
