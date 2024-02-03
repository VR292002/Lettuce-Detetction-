import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

# For the disease
dataset_path = 'E:/Training-Data/train'
num_classes = 13
datagen1 = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator1 = datagen1.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator1 = datagen1.flow_from_directory(dataset_path, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator1, validation_data=val_generator1, epochs=10)
model.save('trained_model1.h5')

# For Lettuce Classification
dataset_path2 = 'E:\Training-Data\Lettuce'
num_classes2 = 5
datagen2 = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator2 = datagen2.flow_from_directory(dataset_path2, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator2 = datagen2.flow_from_directory(dataset_path2, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model2 = Sequential()
model2.add(base_model)
model2.add(GlobalAveragePooling2D())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes2, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.fit(train_generator2, validation_data=val_generator2, epochs=10)
model2.save('trained_model2.h5')

# For Pest Detection
dataset_path3 = 'E:/Training-Data/pest-train-data/train'
num_classes3 = 11
datagen3 = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
train_generator3 = datagen3.flow_from_directory(dataset_path3, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
val_generator3 = datagen3.flow_from_directory(dataset_path3, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model3 = Sequential()
model3.add(base_model)
model3.add(GlobalAveragePooling2D())
model3.add(Dense(128, activation='relu'))
model3.add(Dense(num_classes3, activation='softmax'))
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model3.fit(train_generator3, validation_data=val_generator3, epochs=10)
model3.save('trained_model3.h5')
