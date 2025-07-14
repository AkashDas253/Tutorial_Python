# Tensors in TensorFlow

## About Tensor

### What is a Tensor?
- **Definition**: A tensor is a multi-dimensional array used to represent data in TensorFlow.
- **Dimensions**: Tensors can have different dimensions (also called ranks):
  - **Scalar**: A single number (0-D tensor).
  - **Vector**: A 1-D tensor (e.g., `[1, 2, 3]`).
  - **Matrix**: A 2-D tensor (e.g., `[[1, 2], [3, 4]]`).
  - **Higher-Dimensional Tensors**: Tensors with 3 or more dimensions.

### Importance

- Tensors in TensorFlow are multi-dimensional arrays similar to NumPy arrays but designed to support GPU/TPU computation. 
- TensorFlow provides various functions for tensor creation, manipulation, and operations, enabling deep learning applications and mathematical modeling.

## Creating Tensors

- In TensorFlow, **constants**, **variables**, and **placeholders** are core components for defining and managing data within a computation graph. These elements allow users to create immutable data, modifiable parameters, and placeholders for data that will be fed during runtime, respectively.

### 1. **Constants**

- A constant in TensorFlow is an immutable tensor with a fixed value that cannot be changed after it’s defined. 
- Constants are commonly used for values that should remain the same during computation.

- `tf.constant(value, dtype=None, shape=None, name='Const')`

- **Parameters**:
  - **`value`**: The value to initialize the tensor with (can be a scalar, list, or array-like structure).
  - **`dtype`** (optional, default=`None`): Data type of the output tensor (e.g., `tf.float32`, `tf.int32`). If not specified, it’s inferred from `value`.
  - **`shape`** (optional, default=`None`): Desired shape of the tensor. If specified, the `value` will be broadcast to fill the shape. For example, `shape=(2,3)` creates a 2x3 tensor.
  - **`name`** (optional, default=`'Const'`): A string name for the operation. It’s useful for debugging and visualizing the graph.


- **Notes**:
    - Constants are immutable; they cannot be modified during training or computation.
    - Use constants for data that should remain fixed, like configuration parameters or fixed vectors.

### 2. **Variables**

- Variables in TensorFlow are used to store model parameters and other data that needs to be updated during the training process. 
- Variables can be initialized to any tensor and support dynamic updates.

- `tf.Variable(initial_value, trainable=True, validate_shape=True, caching_device=None, name=None, dtype=None, shape=None, constraint=None, synchronization=tf.VariableSynchronization.AUTO, aggregation=tf.VariableAggregation.NONE)`

- **Parameters**:
  - **`initial_value`**: Initial value of the variable (can be a scalar, list, or a tensor).
  - **`trainable`** (optional, default=`True`): Indicates if the variable should be trainable. When set to `True`, it adds the variable to the list of variables updated during training.
  - **`validate_shape`** (optional, default=`True`): If set to `True`, the shape of the initial value will be verified.
  - **`caching_device`** (optional, default=`None`): Device where the variable should be cached for faster access.
  - **`name`** (optional, default=`None`): A string name for the variable. Useful for debugging and visualizing the graph.
  - **`dtype`** (optional, default=`None`): Data type of the variable. Inferred from `initial_value` if not specified.
  - **`shape`** (optional, default=`None`): Specifies the shape of the variable, which must match `initial_value`.
  - **`constraint`** (optional, default=`None`): Function applied to the variable after updates, e.g., to impose non-negative constraints.
  - **`synchronization`** (optional, default=`tf.VariableSynchronization.AUTO`): Controls when updates are synchronized. Options include `AUTO`, `ON_WRITE`, `ON_READ`.
  - **`aggregation`** (optional, default=`tf.VariableAggregation.NONE`): Defines how values are aggregated. Options include `NONE`, `SUM`, `MEAN`.

- **Notes**:
    - Variables are mutable; their values can be updated during computation.
    - Use variables to define parameters that are optimized during training, such as weights in neural networks.


### 3. **Placeholders** (Note: Placeholders are Deprecated in TensorFlow 2.x)

- In TensorFlow 1.x, **placeholders** allowed users to define inputs that would be provided at runtime. 
- This was particularly useful for defining dynamic inputs for data that would be fed during execution. 
- However, placeholders have been replaced by **`tf.function`** or **`tf.data.Dataset`** in TensorFlow 2.x.

- `tf.compat.v1.placeholder(dtype, shape=None, name=None)`

- **Parameters**:
  - **`dtype`**: Data type of the input to be fed, e.g., `tf.float32`, `tf.int32`.
  - **`shape`** (optional, default=`None`): Shape of the tensor. If `None`, it allows any shape to be fed.
  - **`name`** (optional, default=`None`): A string name for the placeholder.

- **Example** (for TensorFlow 1.x or compatibility mode):
```python
import tensorflow as tf

# Define a placeholder for a float32 tensor
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 3), name="input_placeholder")
```

- In TensorFlow 2.x, placeholders have been replaced with:
    - **`tf.function`**: A decorator that allows using Python functions as part of the TensorFlow computation graph.
    - **`tf.data.Dataset`**: Used for managing data pipelines.

#### **Notes**:
- Placeholders were primarily used in TensorFlow 1.x for feeding data dynamically. They have been deprecated in TensorFlow 2.x, where eager execution is the default.


### Summary Comparison

| Component   | Mutable | Typical Use Case                           | TensorFlow 2.x Alternative                    |
|-------------|---------|--------------------------------------------|-----------------------------------------------|
| **Constant**| No      | Fixed values like configuration constants  | `tf.constant()`                               |
| **Variable**| Yes     | Model parameters (weights, biases)         | `tf.Variable()`                               |
| **Placeholder** (TF 1.x) | No      | Dynamic input for runtime data feeding | `tf.function` and `tf.data.Dataset`           |


### **Generate Tensors**

- `tf.zeros(shape, dtype=tf.float32, name=None)` # Creates a tensor filled with zeros.
- `tf.ones(shape, dtype=tf.float32, name=None)` # Creates a tensor filled with ones.
- `tf.random.uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)` # Generates a tensor with random values from a uniform distribution.

#### **Parameters**:
  - **`shape`**: Defines the shape of the output tensor.
  - **`dtype`** (optional, default=`tf.float32`): Data type of the output tensor (e.g., `tf.float32`, `tf.int32`).
  - **`name`** (optional, default=`None`): Name of the operation (useful for debugging).
  - **`seed`** (optional, default=`None`): Seed for reproducibility.
  - **`minval`** (optional, default=`0`): Minimum value of the random range.
  - **`maxval`** (optional, default=`None`): Maximum value of the random range. If `None`, it defaults to `1` for floats.



### Tensor Properties
- **Shape**: The dimensions of the tensor.
  ```python
  tensor = tf.constant([[1, 2], [3, 4], [5, 6]])
  shape = tensor.shape  # (3, 2)
  ```

- **Data Type**: The type of data stored in the tensor. TensorFlow supports various data types, including:
  - `tf.float16`: 16-bit half-precision floating-point.
  - `tf.float32`: 32-bit single-precision floating-point.
  - `tf.float64`: 64-bit double-precision floating-point.
  - `tf.bfloat16`: 16-bit bfloat16 floating-point.
  - `tf.complex64`: 64-bit single-precision complex.
  - `tf.complex128`: 128-bit double-precision complex.
  - `tf.int8`: 8-bit signed integer.
  - `tf.uint8`: 8-bit unsigned integer.
  - `tf.uint16`: 16-bit unsigned integer.
  - `tf.uint32`: 32-bit unsigned integer.
  - `tf.uint64`: 64-bit unsigned integer.
  - `tf.int16`: 16-bit signed integer.
  - `tf.int32`: 32-bit signed integer.
  - `tf.int64`: 64-bit signed integer.
  - `tf.bool`: Boolean.
  - `tf.string`: String.
  - `tf.qint8`: Quantized 8-bit signed integer.
  - `tf.quint8`: Quantized 8-bit unsigned integer.
  - `tf.qint16`: Quantized 16-bit signed integer.
  - `tf.quint16`: Quantized 16-bit unsigned integer.
  - `tf.qint32`: Quantized 32-bit signed integer.
  - `tf.resource`: Handle to a mutable resource.
  - `tf.variant`: Values of arbitrary types.

  ```python
  dtype = tensor.dtype  # tf.int32
  ```
