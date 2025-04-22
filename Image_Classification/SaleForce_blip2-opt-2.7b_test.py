import os
import glob
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

# Ensure GPU usage if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", force_download=False)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", force_download=False).to(device)

# Specify the folder containing the images
img_dirs = r"C:\Users\77nim\OneDrive\Pictures\Screenshots"
img_extensions = ['jpg', 'jpeg', 'png']

# Function to process a single image and generate a caption
def process_image(img_path):
    try:
        image = Image.open(img_path).convert('RGB')

        # Move image tensor to GPU if available
        inputs = processor(images=image, return_tensors='pt').to(device)

        # Generate a caption for the image
        outputs = model.generate(**inputs, max_new_tokens=25)

        # Decode the generated tokens to text
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        return f"{img_path} : {caption}\n"
    except Exception as e:
        print(f"Error processing the image {img_path}: {e}")
        return None

# Collect all image paths
all_img_paths = [img for ext in img_extensions for img in glob.glob(os.path.join(img_dirs, f"*.{ext}"))]

# Open a file to write the captions
with open("captions_local_images.txt", 'a') as caption_file:
    with ThreadPoolExecutor(max_workers=10) as executor:  # Reduce threads for better GPU handling
        results = list(tqdm(executor.map(process_image, all_img_paths), total=len(all_img_paths), desc="Processing Images"))

    # Write the captions to the file
    for result in results:
        if result is not None:
            caption_file.write(result)