from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
import pprint


###############
# This function returns the relevant information from ONE URL
# Input: one URL
# Output: raw text content
###############
def scrape_one(url):

    loader = AsyncChromiumLoader(url)
    content = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    extracted_content = bs_transformer.transform_documents(content, tags_to_extract=["p", "li", "a", "section"])
    pprint.pprint(extracted_content)

    return extracted_content

url = ["https://www.nhs.uk/conditions/lung-cancer/"]
try:
    extracted_content = scrape_one(url)
except Exception as e:
    print("Error with provided URL")