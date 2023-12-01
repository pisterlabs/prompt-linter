from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper
from langchain.document_loaders import ApifyDatasetLoader

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
# cohere_api_key = os.getenv("COHERE_API_KEY")
apify_api_token = os.getenv("APIFY_API_TOKEN")

# Create an instance of the ApifyWrapper class.
apify = ApifyWrapper()

print("Loading dataset...")

# Run an Apify actor to scrape the data you need.
loader = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://python.langchain.com/en/latest/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] + item["markdown"]or "", metadata={"source": item["url"] , "title" : item["metadata"]["title"] , "description" : item["metadata"]["description"]}
    ),
)
print(loader)

data = loader.load()
data[0]
import cohere
co = cohere.Client('xCRrVzZjuPM5HN6WFKM1eykBBwHMezrMhaQ0AaD7')

response = co.summarize(
  text=data[0],
)
print(response)

# Fetch data from an existing Apify dataset.
# loader = ApifyDatasetLoader(
#     dataset_id="your datasetID",
#     dataset_mapping_function=lambda item: Document(
#         page_content=item["Text"] or "", metadata={"source": item["Link"]}
#     ),
# )

# index = VectorstoreIndexCreator().from_loaders([loader])


# query = "What is Crawlee?"
# result = index.query_with_sources(query)

# # Print the query, answer and sources to the console.
# print(f"Query: {query}")
# print(f"Answer: {result['answer']}")
# print(f"Sources: {result['sources']}")