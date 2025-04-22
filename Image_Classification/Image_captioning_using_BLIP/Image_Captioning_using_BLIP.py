import os
import glob
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor  # Using ThreadPoolExecutor to reduce memory overhead

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

# Use half precision for FP16 to reduce memory usage (if using GPU)
if torch.cuda.is_available():
    model.half()  # Enable half precision for faster processing

model.eval()

# Confirm device
print(f"Model running on: {next(model.parameters()).device}")

# Image directory and supported extensions
img_dirs = r"C:\Users\77nim\OneDrive\Pictures\Screenshots"
img_extensions = ['jpg', 'jpeg', 'png']

# Process a batch of images
def process_images_batch(image_paths):
    images = []
    valid_paths = []

    for img_path in image_paths:
        try:
            images.append(Image.open(img_path).convert("RGB"))
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

    if not images:
        return []

    try:
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=25)

        captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
        return [f"{img_path} : {caption}\n" for img_path, caption in zip(valid_paths, captions)]
    except Exception as e:
        print(f"Error processing batch {valid_paths}: {e}")
        return []

# Gather all image paths
all_img_paths = [
    img for ext in img_extensions
    for img in glob.glob(os.path.join(img_dirs, f"*.{ext}"))
]

# Optional: sort by creation time
all_img_paths.sort(key=os.path.getctime)

# Run captioning if images exist
if not all_img_paths:
    print("No images found in the specified directory.")
else:
    batch_size = 4  # Start with smaller batch size to optimize memory usage
    batches = [all_img_paths[i:i + batch_size] for i in range(0, len(all_img_paths), batch_size)]

    with open("captions_local_images.txt", 'a', encoding='utf-8') as caption_file:
        with ThreadPoolExecutor(max_workers=4) as executor:
            for batch_result in tqdm(executor.map(process_images_batch, batches), total=len(batches), desc="Processing Images"):
                if batch_result:
                    caption_file.writelines(batch_result)
                    print(f"Processed batch:\n{''.join(batch_result)}")
