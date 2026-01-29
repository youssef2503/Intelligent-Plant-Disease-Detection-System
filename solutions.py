treatment_db = {
    "Pepper__bell___Bacterial_spot": "Treatment: Spray with copper-based fungicides. Remove infected leaves immediately to prevent spread.",
    "Pepper__bell___healthy": "Status: Healthy! Keep maintaining regular watering and good soil nutrition.",
    "Tomato_Bacterial_spot": "Treatment: Apply fixed copper bactericides. Avoid overhead watering to keep foliage dry.",
    "Tomato_Early_blight": "Treatment: Use organic fungicides like copper or sulfur. Prune lower leaves to improve airflow.",
    "Tomato_Late_blight": "Treatment: Serious disease! Remove and destroy infected plants immediately. Apply fungicides on remaining plants.",
    "Tomato_Leaf_Mold": "Treatment: Reduce humidity in the greenhouse/garden. Ensure good ventilation and water at the base.",
    "Tomato_Septoria_leaf_spot": "Treatment: Remove fallen leaves. Apply fungicide. Rotate crops next season to prevent recurrence.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Treatment: Spray with water to dislodge mites. Use insecticidal soap or neem oil.",
    "Tomato_Target_Spot": "Treatment: Improve air circulation. Apply fungicides such as chlorothalonil or mancozeb.",
    "Tomato_Yellow_Leaf_Curl_Virus": "Treatment: Control whiteflies using sticky traps. Remove infected plants.",
    "Tomato_Mosaic_virus": "Treatment: Sanitize tools and hands. Remove infected plants. There is no chemical cure.",
    "Tomato__healthy": "Status: Healthy! Your tomato plant is doing great."
}

def get_treatment(class_name):
    return treatment_db.get(class_name, "Unknown Disease or No treatment found in database.")