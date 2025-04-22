import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Filter_images.image_filter import ImageFilter

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained("beingamit99/car_damage_detection")
model = AutoModelForImageClassification.from_pretrained("beingamit99/car_damage_detection")

URL = 'https://www.grays.com/lot/0001-21047473/salvage/2009-holden-commodore-omega-ve-automatic-wagon'

# Load images to a list
images_list = ImageFilter.filter_vehicle_images_from_grays(url=URL)

# Load and process the images

for image in images_list:  # Select the first image from the list
    image = image.convert("RGB")  # Convert to RGB if not already in that mode

    inputs = image_processor(images=image, return_tensors="pt")

    # Make prediction
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()  # Get the logits from the model output
    predicted_class_id = np.argmax(logits)  # Get the predicted class index
    predicted_probability = np.max(logits)  # Get the predicted class probability
    label_map = model.config.id2label  # Get the label map from the model configuration
    predicted_class_name = label_map[predicted_class_id]  # Get the predicted label from the label map

    # Print the results
    #print(f"Predicted class ID: {predicted_class_id}")
    print(f"Predicted class name: {predicted_class_name} probability: {predicted_probability:.2f}")
    #print(f"Predicted class probability: {predicted_probability:.2f}")
    #image.show()  # Display the image