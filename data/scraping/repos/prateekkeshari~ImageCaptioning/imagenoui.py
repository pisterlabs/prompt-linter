import os
from dotenv import load_dotenv
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
import logging

#remove the warning message in terminal
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv() 
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') #replace with your openai api key. Generate a key on https://platform.openai.com/

def collect_image_urls():
    image_urls = []
    while len(image_urls) < 4: #change this number to the number of images you want
        url = input(f"Enter image URL {len(image_urls) + 1} (at least 4 required): ")
        if url:
            image_urls.append(url)
        else:
            print("Please enter a valid image URL.")
    return image_urls

list_image_urls = collect_image_urls()

loader = ImageCaptionLoader(path_images=list_image_urls)
list_docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
while True:
    query = input("Enter your query (type 'exit' to quit): ")
    
    if query.lower() == 'exit':
        break

    result = index.query(query)
    print(result)
