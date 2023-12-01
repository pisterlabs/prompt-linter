from langchain.document_loaders import BrowserlessLoader
import os

token = os.environ["BROWSERLESS_API_TOKEN"]
loader = BrowserlessLoader(
    api_token=token,
    urls=[
        "https://anagora.org/vera",
    ],
    text_content=True,
)

documents = loader.load()

print(documents[0].page_content[:1000])
