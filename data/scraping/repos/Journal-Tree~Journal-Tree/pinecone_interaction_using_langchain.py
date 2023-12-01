# Using more streamlined langchain question-answering API
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
import json
import pinecone


# Open the JSON file
with open('../dataset/student_journal.json') as f:
    # Load the JSON data
    data = json.load(f)
    texts = [Document(page_content=item['entry'], metadata={'text': item['entry']}) for item in data['entries']]

# Access the data
print("Texts loaded")


embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment="us-east-1-aws",  # next to api key in console
)

print("pinecone init")
index_name = "mood-journal-entries"  # Name of the index to create/use
print("creating docsearch")
docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)
print("docsearch created")
# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "melancholic"
print(f"run search:{query}")

docs = docsearch.similarity_search(query)

print(docs[0].page_content)

