from googlesearch import search
from bs4 import BeautifulSoup
import requests
import openai
import config
openai.api_key = config.api_key
def get_image_urls(query, num_results=10):
    image_urls = []

    # Set the search query and number of results you want
    search_query = f'{query} image'

    # Perform the Google image search and retrieve the URLs
    for url in search(search_query, num_results=num_results):
        # Send an HTTP GET request to the search result page
        response = requests.get(url)

        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and extract the direct image URLs from the page
        img_tags = soup.find_all('img')
        for img_tag in img_tags:
            img_url = img_tag.get('src')
            if img_url and img_url.startswith('http'):
                image_urls.append(img_url)

    return image_urls


def getkeyword(notes):
    sections = []
    y = ''
    for note in notes:
        if note != '.':
            y += note
            #print(y+" false")
        else:
            y+='.'
            sections.append(y)
            #print(y+'true')
            y = ''
    return sections      

def makeOneWord(phrase):
    phrase = str(phrase)
    response = openai.Completion.create(
        
        engine="text-davinci-002",  # Use the appropriate engine.
        prompt="Make the following one word, this is for image search purposes: " + phrase,
        max_tokens=50,
        temperature=.3,
    )
    return response['choices'][0]['text']

def getImage(word):
    query = word  # Replace with your search query
    num_results = 5  # Specify the number of image results you want

    image_urls = get_image_urls(query, num_results)

    # Print the direct image URLs
    for i, url in enumerate(image_urls, start=1):
        if word in url:
            return url
        if i == 20:
            return url
    
