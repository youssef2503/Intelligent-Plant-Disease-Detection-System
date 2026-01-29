import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from solutions import get_treatment

MODEL_PATH = 'models/final_finetuned_model.keras'
MAPPING_PATH = 'class_indices.json'
IMG_SIZE = 224

def predict_image(image_path):  
    
    
    with open(MAPPING_PATH, 'r') as f:
        class_names = json.load(f)

    print(f"⏳ Loading Expert Model...")
    model = load_model(MODEL_PATH)

    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0) 


    img_cropped = tf.image.central_crop(img_batch, central_fraction=0.8)
    img_zoomed = tf.image.resize(img_cropped, (IMG_SIZE, IMG_SIZE)) 

    print("Analyzing image (Global view + Detailed view)...")
    pred_global = model.predict(img_batch, verbose=0)  
    pred_zoomed = model.predict(img_zoomed, verbose=0) 

   
    final_score = (pred_global[0] + pred_zoomed[0]) / 2  

    top_indices = final_score.argsort()[-3:][::-1]   
    
    print("\n" + "="*50)
    print(f"🌱 ADVANCED DISEASE REPORT (With TTA)")
    print("="*50)

    top_class = class_names[top_indices[0]] 
    top_conf = final_score[top_indices[0]] * 100 

    print(f"🏆 FINAL PREDICTION:  {top_class}")
    print(f"📊 Confidence:        {top_conf:.2f}%")
    

    print("-" * 30)
    print("🤔 Analysis Breakdown:")
    for i in top_indices:
        name = class_names[i]
        score = final_score[i] * 100
        print(f"   - {name}: {score:.2f}%")
    print("-" * 30)


    if top_conf > 50:
        try:
            treatment = get_treatment(top_class)
            print(f"💊 ADVICE:\n{treatment}")
        except:
            print("No treatment info.")
    else:
        print("⚠️ Warning: Confidence is low. Please use a macro photo of the leaf.")

    print("="*50 + "\n")

if __name__ == "__main__":
    path = input("Enter image path: ").strip('"').strip("'")
    if os.path.exists(path):
        predict_image(path)