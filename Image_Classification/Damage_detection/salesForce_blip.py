import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Filter_images.image_filter import ImageFilter
import torch

images = ImageFilter.filter_vehicle_images_from_grays(url='https://www.grays.com/lot/0001-21047526/motor-vehicles-motor-cycles/2003-toyota-camry-altise-mcv36r-automatic-sedan')

model_name = "Salesforce/blip-vqa-capfilt-large"

processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
model = BlipForQuestionAnswering.from_pretrained(model_name).to(device="cuda" if torch.cuda.is_available() else "cpu")

questions = [
    "Check carefully and tell me is there any scratches in any panel of the car?",
    "Check carefully and tell me is there any minor dents in any panel of the car?",
    "Check carefully and tell me is there any medium dents in any panel of the car?",
    "Check carefully and tell me is there any large dents in any panel of the car?",
    "Check carefully and tell me what is the overall condition of the body of the car showed in the image?",
]
q1 = {}
q2 = {}
q3 = {}
q4 = {}
q5 = {}

for q, question in enumerate(questions):
    for i, image in enumerate(images[:4], start=1):  # Start from 1 for better readability
        image = image.convert("RGB")  # Convert to RGB if not already in that mode
        inputs = processor(image, question, return_tensors="pt").to(device="cuda" if torch.cuda.is_available() else "cpu")
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        #print(f"Image : {i}:")
        #print(f"Question: {question}, \t Answer: {answer} \n")
        if q == 0:
            q1.update({i: answer})
        elif q == 1:   
            q2.update({i: answer})
        elif q == 2:
            q3.update({i: answer})
        elif q == 3:
            q4.update({i: answer})
        elif q == 4:
            q5.update({i: answer}) 


print("Q1: Check carefully and tell me is there any scratches in the car?")
for i, answer in q1.items():
    print(f"Image {i}: {answer}")  

print("\nQ2: Check carefully and tell me is there any minor dents in the car?")
for i, answer in q2.items():
    print(f"Image {i}: {answer}")

print("\nQ3: Check carefully and tell me is there any medium dents in the car?")
for i, answer in q3.items():
    print(f"Image {i}: {answer}")

print("\nQ4: Check carefully and tell me is there any large dents in the car?")
for i, answer in q4.items():
    print(f"Image {i}: {answer}")

print("\nQ5: Check carefully and tell me what is the overall condition of the body of the car showed in the image?")
for i, answer in q5.items():
    print(f"Image {i}: {answer}")



