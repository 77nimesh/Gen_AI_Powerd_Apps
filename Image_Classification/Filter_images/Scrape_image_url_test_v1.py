from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def fetch_and_filter_images(url, min_size=(400, 400)):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch page: {e}")
        return []

    print(f"Status: {response.status_code}, Content-Type: {response.headers.get('Content-Type')}")

    soup = BeautifulSoup(response.text, 'html.parser')
    div_ribbons = soup.find_all('div', class_='dvRibbon')

    image_links = []
    for href in div_ribbons:
        a_tag = href.find('a')
        if a_tag and a_tag.get('href'):
            image_links.append(a_tag['href'])

    image_links = list(set(image_links))  # Remove duplicates

    filtered_images = []

    for link in tqdm(image_links, desc="Downloading Images"):
        try:
            page = requests.get(link, headers=headers, timeout=10)
            if page.status_code == 200:
                img = Image.open(BytesIO(page.content))
                img.verify()
                img = Image.open(BytesIO(page.content))  # Reopen after verify
                if img.size[0] > min_size[0] and img.size[1] > min_size[1]:
                    filtered_images.append(img)
        except Exception as e:
            print(f"Error processing {link}: {e}")
            continue

    return filtered_images


if __name__ == '__main__':
    URL = 'https://www.grays.com/lot/0001-50514338/motor-vehicles-motor-cycles/2017-mitsubishi-lancer-es-sport-cf-cvt-sedan'
    images = fetch_and_filter_images(URL)
    print(f"Total images filtered: {len(images)}")
