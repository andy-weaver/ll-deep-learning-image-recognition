"""Model to recognize images from CIFAR-10 dataset using Keras."""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    Dense,
    MaxPooling2D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from pathlib import Path

# Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize image data to be in (0, 1) interval
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

# Convert class labels to one-hot encoded vectors
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Create a model
model = Sequential()

# Add convolutional layers
model.add(
    Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3), padding="same")
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# More convolutional layers
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Add dense layers
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

# Print model summary
model.summary()

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Load model weights
model.load_weights("model.weights.h5")

# Train the model
model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=30,
    validation_data=(X_test, y_test),
    shuffle=True,
)

# Save the model structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save the model's trained weights
model.save_weights("model.weights.h5")