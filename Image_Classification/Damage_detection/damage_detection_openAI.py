import os
import openai
from PIL import Image
from io import BytesIO 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Filter_images.image_filter import ImageFilter

openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a function to classify the image using OpenAI's API
def classify_image(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = openai.Image.create(
        file=image_data,
        model="image-alpha-001",
        n=1,
        size="1024x1024",
        response_format="url"
    )

    return response["data"][0]["url"]  # Return the URL of the classified image

classify_image(r"C:\Users\77nim\Downloads\OIP.jpeg")  # Replace with your image path