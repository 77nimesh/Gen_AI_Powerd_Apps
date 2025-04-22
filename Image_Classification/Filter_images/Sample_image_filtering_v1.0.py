from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

url = 'https://www.grays.com/lot/0001-50514338/motor-vehicles-motor-cycles/2017-mitsubishi-lancer-es-sport-cf-cvt-sedan'

response = requests.get(url)

print(response.status_code)  # Check if the request was successful (200 OK)
print(response.headers['Content-Type'])  # Check the content type of the response

soup = BeautifulSoup(response.text, 'html.parser')

#print(soup)  # Print the prettified HTML content of the page

#print(soup.prettify())  # Print the prettified HTML content of the page

#print(soup.find_all('script'))  # Find all script tags in the HTML
#print(soup.find_all('link'))  # Find all link tags in the HTML  
#print(soup.find_all('scr'))  # Find all meta tags in the HTML

divRibbon = soup.find_all('div', class_='dvRibbon') # Find all div tags with class "dvRibbon"

image_links = []

for href in divRibbon:
    link = href.find('a')['href']  # Find the 'href' attribute of the 'a' tag within the div
    image_links.append(link)  # Append the link to the list

filtered_links = []
for link in image_links:
    page = requests.get(link)
    if page.status_code == 200:
        img = Image.open(BytesIO(page.content)) 
        if img.size > (400, 400):
            filtered_links.append(link)
        else:
            continue

for link in filtered_links:
    print(link,'\n')  # Print the filtered image links

print(len(filtered_links))  # Print the number of filtered image links