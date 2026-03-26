import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_disease_efficientnet.keras")

# Class names
class_names = [
"Pepper__bell___Bacterial_spot",
"Pepper__bell___healthy",
"Potato___Early_blight",
"Potato___Late_blight",
"Potato___healthy",
"Tomato_Bacterial_spot",
"Tomato_Early_blight",
"Tomato_Late_blight",
"Tomato_Leaf_Mold",
"Tomato_Septoria_leaf_spot",
"Tomato_Spider_mites",
"Tomato_Target_Spot",
"Tomato_Yellow_Leaf_Curl_Virus",
"Tomato_mosaic_virus",
"Tomato_healthy"
]

# Load image
image_path = "/Users/nischalmittal/Downloads/FINAL_FYP_ME/test_leaf.png"

img = Image.open(image_path).convert("RGB").resize((224,224))
img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img).squeeze()

max_probability = np.max(prediction)
class_index = np.argmax(prediction)

# Confidence check
confidence = float(max_probability)

if confidence < 0.6:
    result = "Unknown Disease"
else:
    result = class_names[class_index]

print("\nPrediction:", result)
print("Class Index:", class_index)
print("Confidence:", round(confidence*100,2), "%")