# TensorFlow Cheatsheet

## 1. Importing TensorFlow
- import tensorflow as tf

## 2. Creating Tensors
- tf.constant(value)  # Create a constant tensor
- tf.zeros(shape)  # Create a tensor filled with zeros
- tf.ones(shape)  # Create a tensor filled with ones
- tf.random.normal(shape)  # Create a tensor with random normal values

## 3. Tensor Operations
- tf.add(tensor1, tensor2)  # Element-wise addition
- tf.multiply(tensor1, tensor2)  # Element-wise multiplication
- tf.matmul(tensor1, tensor2)  # Matrix multiplication
- tf.reduce_mean(tensor)  # Mean of tensor elements
- tf.reduce_sum(tensor)  # Sum of tensor elements

## 4. Reshaping Tensors
- tf.reshape(tensor, new_shape)  # Reshape tensor
- tf.transpose(tensor)  # Transpose tensor

## 5. Building Models (Sequential API)
- from tensorflow.keras.models import Sequential  # Import Sequential model
- model = Sequential()  # Initialize model
- model.add(tf.keras.layers.Dense(units, activation='relu', input_shape=(input_shape,)))  # Add Dense layer

## 6. Compiling Models
- model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Compile model

## 7. Fitting Models
- model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Train model

## 8. Evaluating Models
- loss, accuracy = model.evaluate(X_test, y_test)  # Evaluate model on test data

## 9. Making Predictions
- predictions = model.predict(X_new)  # Predict on new data
- predicted_classes = tf.argmax(predictions, axis=1)  # Get predicted class labels

## 10. Saving and Loading Models
- model.save('model.h5')  # Save model
- from tensorflow.keras.models import load_model  # Load model
- model = load_model('model.h5')  # Load saved model

## 11. Callbacks
- from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Import callbacks
- early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Early stopping
- model.fit(X_train, y_train, callbacks=[early_stopping])  # Fit with callbacks

## 12. Custom Training Loop
- for epoch in range(epochs):  # Custom training loop
  - with tf.GradientTape() as tape:  # Record gradients
    - predictions = model(X_batch)  # Forward pass
    - loss = loss_fn(y_batch, predictions)  # Compute loss
  - gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
  - optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update weights

## 13. Data Pipelines
- from tensorflow.data import Dataset  # Import Dataset
- dataset = Dataset.from_tensor_slices((X, y)).batch(32)  # Create dataset and batch

## 14. Data Augmentation
- from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Import ImageDataGenerator
- datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2)  # Create data generator
- datagen.flow(X_train, y_train, batch_size=32)  # Flow data

## 15. Transfer Learning
- from tensorflow.keras.applications import VGG16  # Import pre-trained model
- base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Load VGG16
- model = Sequential([base_model, tf.keras.layers.Flatten(), tf.keras.layers.Dense(units, activation='softmax')])  # Create new model
