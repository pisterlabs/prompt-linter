import sys
import pandas as pd
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# Specify the default API key for OpenAI embeddings
DEFAULT_OPENAI_API_KEY = "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"

# Get the command line arguments
args = dict(arg.split('=') for arg in sys.argv[1:])
csv_path = args.get("--prod-csv", "data/t-2.csv")
prod_embedding_store = args.get("--prod-embedding-store", 'prod-embeddings-store')
openai_api_key = args.get("--openai-api-key", DEFAULT_OPENAI_API_KEY)

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Generate a list of Documents
documents = []
data_list = df.to_dict(orient='records')
for item in data_list:
    page_content = ' '.join([item['category-links'], item['subcategory-links'], item['name']])
    document = Document(page_content=page_content, metadata=item)
    documents.append(document)

# Get your embeddings engine ready
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Embed your documents and combine with the raw text in a pseudo db. Note: This will make an API call to OpenAI
docsearch = FAISS.from_documents(documents, embeddings)

# Example search
query = 'Asus laptop'
found_docs = docsearch.similarity_search('Asus laptop')
print(f'try query: "{query}" and get results: {found_docs}')

# Save the vectorstore to a local directory
docsearch.save_local(prod_embedding_store)
