### Basics of TensorFlow: A Comprehensive Guide Covering 80-90% of Core Concepts

TensorFlow is an open-source deep learning framework developed by Google. It provides a flexible ecosystem for building and deploying machine learning models, from simple computations to complex neural networks. Below is a detailed overview of the essential aspects of TensorFlow that cover the most commonly used concepts.

---

### 1. **Core Concepts**

#### **1.1 Tensors**
- A tensor is a multi-dimensional array (similar to NumPy arrays) that forms the foundation of TensorFlow computations.
- **Syntax:**
  ```python
  import tensorflow as tf

  # Scalar
  scalar = tf.constant(5)

  # Vector
  vector = tf.constant([1, 2, 3])

  # Matrix
  matrix = tf.constant([[1, 2], [3, 4]])

  # Tensor
  tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  ```

#### **1.2 Computational Graph**
- A computational graph is a directed graph where nodes represent operations, and edges represent tensors flowing between them.
- TensorFlow automatically builds and manages this graph.

#### **1.3 Eager Execution**
- Enabled by default, allowing operations to execute immediately (imperative programming).
- **Syntax:**
  ```python
  tf.executing_eagerly()  # Returns True
  ```

---

### 2. **Data Handling**

#### **2.1 Tensor Creation**
- **Common Tensor Creation Functions:**
  ```python
  zeros = tf.zeros([2, 3])           # Tensor of zeros
  ones = tf.ones([2, 3])             # Tensor of ones
  random = tf.random.uniform([2, 3]) # Random uniform distribution
  ```
- **Reshape Tensors:**
  ```python
  reshaped_tensor = tf.reshape(tensor, [2, -1])  # Automatically infers second dimension
  ```

#### **2.2 Datasets**
- Use `tf.data` for input pipelines.
- **Syntax:**
  ```python
  data = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  data = data.map(lambda x: x * 2).batch(2)
  for element in data:
      print(element)
  ```

---

### 3. **Building Models**

#### **3.1 Layers**
- Layers are the building blocks of neural networks.
- **Syntax:**
  ```python
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  ```

#### **3.2 Model Compilation**
- Specifies the optimizer, loss, and metrics for the training process.
- **Syntax:**
  ```python
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ```

#### **3.3 Model Training**
- Use the `fit` method to train the model.
- **Syntax:**
  ```python
  model.fit(x_train, y_train, epochs=5, batch_size=32)
  ```

#### **3.4 Model Evaluation**
- Use `evaluate` to test model performance.
- **Syntax:**
  ```python
  loss, accuracy = model.evaluate(x_test, y_test)
  ```

---

### 4. **Activation Functions**

Common activation functions include:
- **ReLU**: `tf.keras.activations.relu(x)`
- **Sigmoid**: `tf.keras.activations.sigmoid(x)`
- **Softmax**: `tf.keras.activations.softmax(x)`

---

### 5. **Optimizers**

Common optimizers:
- **SGD**: `tf.keras.optimizers.SGD(learning_rate=0.01)`
- **Adam**: `tf.keras.optimizers.Adam(learning_rate=0.001)`

---

### 6. **Loss Functions**

Loss functions measure the error between predictions and actual values:
- **Mean Squared Error**: `tf.keras.losses.MeanSquaredError()`
- **Categorical Crossentropy**: `tf.keras.losses.CategoricalCrossentropy()`

---

### 7. **Custom Training Loop**
- TensorFlow allows custom training loops for advanced use cases.
- **Syntax:**
  ```python
  for epoch in range(epochs):
      for x, y in dataset:
          with tf.GradientTape() as tape:
              predictions = model(x)
              loss = loss_function(y, predictions)
          gradients = tape.gradient(loss, model.trainable_variables)
          optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  ```

---

### 8. **Saving and Loading Models**

#### **Saving a Model**
- **Syntax:**
  ```python
  model.save('model_path')
  ```

#### **Loading a Model**
- **Syntax:**
  ```python
  loaded_model = tf.keras.models.load_model('model_path')
  ```

---

### 9. **TensorFlow Utilities**

#### **Math Operations**
- **Syntax:**
  ```python
  tf.add(x, y)
  tf.multiply(x, y)
  tf.reduce_mean(tensor)
  ```

#### **Gradient Calculation**
- **Syntax:**
  ```python
  with tf.GradientTape() as tape:
      y = x ** 2
  gradient = tape.gradient(y, x)
  ```

---

### 10. **High-Level APIs**

#### **Keras API**
- Provides a high-level API for building and training models.
- **Syntax:**
  ```python
  from tensorflow.keras import layers, models
  model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
  ])
  ```

#### **TFHub**
- TensorFlow Hub allows the use of pre-trained models.
- **Syntax:**
  ```python
  import tensorflow_hub as hub
  model = tf.keras.Sequential([hub.KerasLayer('url_to_model')])
  ```

---

### 11. **Distributed Computing**
- Use `tf.distribute.Strategy` for distributed training across devices.
- **Syntax:**
  ```python
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
      model = tf.keras.Sequential([...])
  ```

---

### Conclusion

This guide covers **80-90%** of TensorFlow's core functionality:
- Working with tensors and datasets.
- Building and training machine learning models.
- Using common layers, loss functions, optimizers, and activation functions.
- Saving/loading models and performing distributed training.

These concepts form the foundation for most TensorFlow projects, whether simple or complex.