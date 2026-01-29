import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

DATASET_PATH = 'dataset'
IMG_SIZE = 224
BATCH_SIZE = 32 
EPOCHS = 20  

print("⏳ Loading Dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE 
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
n_classes = len(class_names) 




data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
]) 

print("🧠 Setting up MobileNetV2 for Fine-Tuning...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False, 
    weights='imagenet'
)

base_model.trainable = True 

print(f"Number of layers in the base model: {len(base_model.layers)}")
fine_tune_at = 100 

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False  

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False) 
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(n_classes, activation='softmax')(x)  

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),  
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    'models/final_finetuned_model.keras',
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True 
)


print("🚀 Starting Fine-Tuning (This creates the Expert Model)...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds, 
    callbacks=[checkpoint, early_stopping]
)

print("💾 Fine-Tuned Model Saved. Evaluating...")
loss, acc = model.evaluate(val_ds)
print(f"🌟 Final Accuracy after Fine-Tuning: {acc*100:.2f}%")