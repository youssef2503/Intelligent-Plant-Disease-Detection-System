import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATASET_PATH = 'dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = 'models/final_finetuned_model.keras'

print("⏳ Loading Dataset for Evaluation...")
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
print(f"✅ Classes: {class_names}")

def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(1000, seed=12)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    test_ds = ds.skip(train_size).skip(val_size)
    return test_ds

test_ds = get_dataset_partitions_tf(dataset)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

if not os.path.exists(MODEL_PATH):
    print("❌ Error: Model file not found!")
    exit()

print("⏳ Loading Model...")
model = load_model(MODEL_PATH)


print("\n" + "="*40)
print("🧪 FINAL EVALUATION ON TEST SET")
print("="*40)
test_loss, test_acc = model.evaluate(test_ds)
print(f"✅ Final Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Final Test Loss: {test_loss:.4f}")
print("="*40 + "\n")

print("📊 Generating Confusion Matrix...")
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

if not os.path.exists('outputs'): os.makedirs('outputs')
plt.savefig('outputs/confusion_matrix_final.png')
print("🎉 DONE! Confusion matrix saved to 'outputs/confusion_matrix_final.png'.") 