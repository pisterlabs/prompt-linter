import os
import config
from datetime import datetime
import langchain.document_loaders
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

# Directory to save articles
ARTICLES_DIR = 'articles2'
if not os.path.exists(ARTICLES_DIR):
    os.makedirs(ARTICLES_DIR)

# Load online PDF
file_path = "https://s201.q4cdn.com/254090064/files/doc_presentations/2023/Sep/19/2023-09-21-Rochester-Site-Tour-Final.pdf"
loader = langchain.document_loaders.OnlinePDFLoader(file_path)

# Load and split local PDF
local_dir = r"C:\Users\Robert Walter\Dropbox\mining\CDE\2022-Annual-Report-PDF.pdf"
loaderLocalDir = langchain.document_loaders.PyPDFLoader(local_dir)
pages = loaderLocalDir.load_and_split()

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(pages)

# Vectorize and store
vector_store = Chroma.from_documents(docs[:166], OpenAIEmbeddings())
if len(docs) > 166:
    for i in range(166, len(docs), 166):
        batch = docs[i:i+166]
        vector_store.add_documents(batch)

# Process user queries
while True:
    query = input("What would you like to ask? (type 'quit' to exit) ")
    if query.lower() == 'quit':
        break

    response = vector_store.similarity_search(f"{query} ?", k=1)

    for item in response:
        page_content = item.page_content
        metadata = item.metadata
        page_number = metadata.get('page', 'N/A')
        source = metadata.get('source', 'N/A')

        # Format the text
        page_content = page_content.replace('\n', '\r\n')
        page_content = ' '.join(page_content.split())

        # Create a filename
        safe_query = query.replace(" ", "_").replace("?", "").replace("/", "_")
        safe_source = source.replace(" ", "_").replace("?", "").replace("/", "_").replace("\\", "_")
        save_file = f"{ARTICLES_DIR}/{safe_query}-{page_number}-{safe_source}.txt"

        # Save to file
        with open(save_file, 'w', encoding='utf-8') as file:
            file.write(page_content)
            file.flush()
            os.fsync(file.fileno())

        print(f"Data written to {save_file}: {len(page_content)} characters")
        print(page_content)
        print("--------------------------------------------------")

