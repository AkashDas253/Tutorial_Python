# Keras Cheatsheet

## 1. Importing Keras
- from keras.models import Sequential
- from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

## 2. Creating a Model
- model = Sequential()  # Initialize model

## 3. Adding Layers
- model.add(Dense(units, activation='relu', input_dim=input_shape))  # Fully connected layer
- model.add(Dropout(rate))  # Dropout layer
- model.add(Flatten())  # Flatten input
- model.add(Conv2D(filters, kernel_size, activation='relu', input_shape=(height, width, channels)))  # Convolutional layer
- model.add(MaxPooling2D(pool_size))  # Max pooling layer

## 4. Compiling the Model
- model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model

## 5. Fitting the Model
- model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  # Train model

## 6. Evaluating the Model
- loss, accuracy = model.evaluate(X_test, y_test)  # Evaluate model on test data

## 7. Making Predictions
- predictions = model.predict(X_new)  # Predict on new data
- predicted_classes = np.argmax(predictions, axis=1)  # Get predicted class labels

## 8. Saving and Loading Models
- model.save('model.h5')  # Save model
- from keras.models import load_model  # Load model
- model = load_model('model.h5')  # Load saved model

## 9. Customizing Training Loop
- for epoch in range(epochs):  # Custom training loop
  - model.fit(X_batch, y_batch)  # Fit on batch
  - model.evaluate(X_val, y_val)  # Evaluate on validation set

## 10. Callbacks
- from keras.callbacks import EarlyStopping, ModelCheckpoint  # Import callbacks
- early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # Early stopping
- model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)  # Save best model
- model.fit(X_train, y_train, callbacks=[early_stopping, model_checkpoint])  # Fit with callbacks

## 11. Data Preprocessing
- from keras.preprocessing.image import ImageDataGenerator  # Image data generator
- datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2)  # Data augmentation
- datagen.flow(X_train, y_train, batch_size=32)  # Flow data

## 12. Transfer Learning
- from keras.applications import VGG16  # Import pre-trained model
- base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Load VGG16
- model = Sequential()  # Create new model
- model.add(base_model)  # Add pre-trained model
- model.add(Flatten())  # Flatten
- model.add(Dense(units, activation='softmax'))  # Add output layer
