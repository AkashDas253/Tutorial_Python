
## Tensor Operations

### **Basic Operations**: 

- `result = tf.add(tensor1, tensor2)` or simply `result = tensor1 + tensor2`
- `result = tf.subtract(tensor1, tensor2)` or `result = tensor1 - tensor2`
- `result = tf.multiply(tensor1, tensor2)` or `result = tensor1 * tensor2`
- `result = tf.divide(tensor1, tensor2)` or `result = tensor1 / tensor2`

```python
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = tf.add(a, b)  # [4, 6]
d = tf.multiply(a, b)  # [3, 8]
```

### **Matrix Operations**
    
- `tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)` # Performs matrix multiplication between two tensors.

#### **Parameters**:
- **`a`**: First tensor.
- **`b`**: Second tensor.
- **`transpose_a`** (optional, default=`False`): Transposes `a` before multiplication.
- **`transpose_b`** (optional, default=`False`): Transposes `b` before multiplication.
- **`adjoint_a`** and **`adjoint_b`** (optional, default=`False`): Conjugate transpose of `a` and `b`.
- **`a_is_sparse`** and **`b_is_sparse`** (optional, default=`False`): Treats tensors as sparse for optimization.
- **`name`** (optional, default=`None`): Operation name.

```python
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])
product = tf.matmul(matrix1, matrix2)  # [[19, 22], [43, 50]]
```

### **Aggregation Operations**

- `tf.reduce_sum(input_tensor, axis=None, keepdims=False, name=None)` # Computes the sum of elements across dimensions.
- `tf.reduce_mean(input_tensor, axis=None, keepdims=False, name=None)`  # Calculates the mean of elements across dimensions.

#### **Parameters**:
- **`input_tensor`**: Input tensor.
- **`axis`** (optional, default=`None`): Axis to reduce along.
- **`keepdims`** (optional, default=`False`): If `True`, retains reduced dimensions with size 1.
- **`name`** (optional, default=`None`): Operation name.

### **Indexing and Slicing**

- TensorFlow allows for indexing and slicing tensors, similar to NumPy.

#### **Example**:
```python
x = tf.constant([[1, 2], [3, 4], [5, 6]])
print(x[1])       # Second row: [3, 4]
print(x[:, 0])    # First column: [1, 3, 5]
```

### Reshaping Tensors

- `tf.reshape(tensor, shape, name=None)` # Reshapes a tensor to a specified shape.

#### **Parameters**:
- **`tensor`**: Input tensor to be reshaped.
- **`shape`**: Target shape, defined as a list or tuple.
- **`name`** (optional, default=`None`): Name of the operation.
#### **Examle**
```python
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
reshaped_tensor = tf.reshape(tensor, [3, 2])  # [[1, 2], [3, 4], [5, 6]]
```

### **Comparison Operations**

- `tf.greater(x, y)`, `tf.less(x, y)`, `tf.equal(x, y, name=None)`

#### **Example**:
```python
a = tf.constant([1, 2, 3])
b = tf.constant([1, 0, 3])
result = tf.equal(a, b)  # Output: [True, False, True]
```



### **Automatic Differentiation**

- TensorFlow’s `tf.GradientTape` enables automatic differentiation for computing gradients.

- `tf.GradientTape(persistent=False, watch_accessed_variables=True)`

- Context manager that records operations for automatic differentiation.

#### **Parameters**:
- **`persistent`** (optional, default=`False`): If `True`, allows multiple gradient calculations.
- **`watch_accessed_variables`** (optional, default=`True`): Tracks variables accessed within the tape context.

#### **Example**:

```python
x = tf.Variable([2.0, 3.0])

with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)
print(grad)  # Output: [4.0, 6.0]
```

### **TensorFlow Eager Execution**

- **Eager Execution**: An imperative programming environment that evaluates operations immediately.
  
  ```python
  tf.config.run_functions_eagerly(True)
  ```


### **Note**

- TensorFlow operations are executed within a computational graph. Ensure that you have initialized the necessary components (e.g., TensorFlow session in TensorFlow 1.x or eager execution in TensorFlow 2.x).
- TensorFlow 2.x uses eager execution by default, which means operations are evaluated immediately.
- `tf.GradientTape` is useful for automatic differentiation, especially in training machine learning models.


### Summary
- Tensors are the core data structures in TensorFlow.
- They can be created as constants or variables.
- TensorFlow supports various operations on tensors, including basic arithmetic and matrix operations.
- Tensors have properties like shape and data type.
- Tensors can be reshaped to different dimensions.
- Eager execution allows for immediate evaluation of operations.

### **Comprehensive Example**

```python
import tensorflow as tf

# Basic Operations
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = tf.add(a, b)  # [4, 6]
d = tf.multiply(a, b)  # [3, 8]

# Reshape Operation
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
reshaped = tf.reshape(tensor, [3, 2])  # [[1, 2], [3, 4], [5, 6]]

# Matrix Operations
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])
product = tf.matmul(matrix1, matrix2)  # [[19, 22], [43, 50]]

# Aggregation Operations
sum_result = tf.reduce_sum(tensor, axis=0)  # [5, 7, 9]
mean_result = tf.reduce_mean(tensor, axis=1)  # [2, 5]

# Indexing and Slicing
x = tf.constant([[1, 2], [3, 4], [5, 6]])
second_row = x[1]  # [3, 4]
first_column = x[:, 0]  # [1, 3, 5]

# Comparison Operations
a = tf.constant([1, 2, 3])
b = tf.constant([1, 0, 3])
comparison_result = tf.equal(a, b)  # [True, False, True]

# Automatic Differentiation
x = tf.Variable([2.0, 3.0])
with tf.GradientTape() as tape:
    y = x ** 2
grad = tape.gradient(y, x)  # [4.0, 6.0]

# Eager Execution
tf.config.run_functions_eagerly(True)

# Print results
print("Basic Operations:", c.numpy(), d.numpy())
print("Reshape Operation:", reshaped.numpy())
print("Matrix Operations:", product.numpy())
print("Aggregation Operations:", sum_result.numpy(), mean_result.numpy())
print("Indexing and Slicing:", second_row.numpy(), first_column.numpy())
print("Comparison Operations:", comparison_result.numpy())
print("Automatic Differentiation:", grad.numpy())
```