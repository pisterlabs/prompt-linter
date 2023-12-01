import os
from dotenv import load_dotenv
from langchain.document_loaders import ImageCaptionLoader
from langchain.indexes import VectorstoreIndexCreator
import logging

#remove the warning message in terminal
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("tokenizers").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

#load the .env file
load_dotenv()

#replace with your openai api key. Generate a key on https://platform.openai.com/
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# image_urls = ['archie_and_trixie.jpg']
# image_urls = ['kev.jpg']
# image_urls = ['kev2.jpg']
image_urls = ['archie.jpg']
# image_urls = ['frankie.jpg']

# you can provide more than one image
# image_urls = ['frankie.jpg']

loader = ImageCaptionLoader(path_images=image_urls)
list_docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])

# result = index.query('describe what is in the image, be as descriptive as possible using poetic language')
result = index.query('describe what is in the image, be nonchalant and snarky')
print(result)