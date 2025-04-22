from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load the model and processor
model_name = "google/vit-base-patch16-224"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

path = r"C:\Users\77nim\Downloads\43989448.jpg"

# Load and preprocess the image
image = Image.open(path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class]}")