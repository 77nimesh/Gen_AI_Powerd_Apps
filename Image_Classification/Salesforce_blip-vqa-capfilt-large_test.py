import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to("cuda")

img_url = "https://res0.grays.com/handlers/imagehandler.ashx?t=sh&id=43989448&s=d&index=0&ts=638804138770000000" 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "Is there any dents in the car?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_length=16, min_length=2, num_beams=5, do_sample=False, early_stopping=False)
print(processor.decode(out[0], skip_special_tokens=True))
