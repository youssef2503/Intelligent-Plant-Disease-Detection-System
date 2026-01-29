import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

DATASET_PATH = 'dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

print(" Loading Dataset...")
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(f" Classes found: {class_names}")

print("⚖️ Calculating Class Weights...")
class_counts = {}
total_samples = 0

for i, class_name in enumerate(class_names):
    path = os.path.join(DATASET_PATH, class_name)
    if os.path.exists(path):
        count = len(os.listdir(path))
    else:
        count = 0
    class_counts[i] = count
    total_samples += count
    print(f"   - {class_name}: {count} images")

class_weights = {}
n_classes = len(class_names)
for i in range(n_classes):
    if class_counts[i] > 0:
        weight = total_samples / (n_classes * class_counts[i])
        class_weights[i] = weight
    else:
        class_weights[i] = 1.0

print(f"Final Weights: {class_weights}")

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(1000, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255),
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
])

input_shape = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(n_classes, activation='softmax')
])

model.build(input_shape=input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


checkpoint = ModelCheckpoint(
    'models/my_plant_model.keras',
    monitor='val_accuracy',  
    save_best_only=True,     
    mode='max',            
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,              
    restore_best_weights=True
)

print(" Starting Training...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stopping]
)

if not os.path.exists('models'): os.makedirs('models')
model.save('models/my_plant_model.keras')
print(" Model Saved Successfully!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

if not os.path.exists('outputs'): os.makedirs('outputs')
plt.savefig('outputs/accuracy_plot.png')

print(" Generating Confusion Matrix...")
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')
print(" DONE! Check outputs folder.")
