import os

from langchain.retrievers import AzureCognitiveSearchRetriever

os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "positive-c-search"
os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"] = "azureblob-index"
os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"] = ""

retriever = AzureCognitiveSearchRetriever(content_key="content", top_k=3)

upit = input("Unesite upit: ")

odgovor = retriever.get_relevant_documents(upit)

print(odgovor[0].page_content.strip())
