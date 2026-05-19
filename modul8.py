# ============================================================
# CELL 1 - Import library
# ============================================================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# CELL 2 - ImageDataGenerator
# ============================================================
dataset_path = "./rockpaperscissors"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)


# ============================================================
# CELL 3 - Membuat Model CNN
# ============================================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])
model.summary()


# ============================================================
# CELL 4 - Compile Model
# ============================================================
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ============================================================
# CELL 5 - Training Model
# ============================================================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

# ============================================================
# CELL 6 - Evaluasi Model
# ============================================================
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')


# ============================================================
# CELL 7 - Prediksi
# ============================================================
predictions = model.predict(validation_generator)
print(predictions)