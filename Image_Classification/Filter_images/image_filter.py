import random
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

class ImageFilter:
    @staticmethod
    def filter_vehicle_images_from_grays(url, min_size=(400,400)):
        """
        Filters vehicle images from Grays website based on a minimum size.

        Parameters:
        url (str): The URL of the Grays page to scrape.
        min_size (tuple): Minimum size for filtering images (width, height).

        Returns:
        list: A list of filtered PIL Image objects.
        """
        user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.137 Safari/537.36',
        ]
        headers = {'User-Agent': random.choice(user_agents)}

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()  # Raise an error for bad responses
        except requests.RequestException as e:  
            print(f"Error fetching the URL: {e}")
            return []       
        
        print(f'Status Code: {response.status_code}')
        print(f'Content Type: {response.headers.get("Content-Type", "Unknown")}')
        
        soup = BeautifulSoup(response.content, 'html.parser')
        div_ribbon = soup.find_all('div', class_='dvRibbon')

        images_list = []
        try:
            for href in tqdm(div_ribbon, desc="Processing images", unit="image"):
                link = href.find('a')['href']
                img_data = requests.get(link, headers=headers, timeout=30)
                if img_data.status_code == 200:
                    try:
                        img = Image.open(BytesIO(img_data.content))
                        img.verify()  # Verify that the image is valid
                        img = Image.open(BytesIO(img_data.content))  # Reopen the image to get its size
                        if img.size[0] >= min_size[0] and img.size[1] >= min_size[1]:
                            images_list.append(img)
                        else:
                            continue
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        continue
                else:
                    print(f"Failed to fetch image from {link}: {img_data.status_code}")
        except Exception as e:
            print(f"Error processing div_ribbon: {e}")  

        return images_list