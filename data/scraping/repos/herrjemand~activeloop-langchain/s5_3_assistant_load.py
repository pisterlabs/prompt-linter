from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import os, requests, re
from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
import chromadb, uuid

def scrape_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.body.text.strip()

    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def scrape_all_content(paths, output_file):
    print("Scraping websites...")
    content = []
    for doc_path in paths:
        scraped_content = scrape_page_content(f"https://huggingface.co{doc_path}")
        content.append(scraped_content.rstrip("\n"))

    with open(output_file, "w") as f:
        for item in content:
            f.write("%s\n" % item)

    return content

def load_docs(root_dir, filename):
    print("Loading documents...")
    docs = []

    try:
        loader = TextLoader(os.path.join(root_dir, filename), encoding="utf-8")

        docs.extend(loader.load_and_split())
    except Exception as e:
        print(e)
        pass

    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(docs)

    return docs

chat_model = ChatOpenAI(model_name="davinci")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

chroma_dataset_name = "assistant_53"
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
chroma_client.delete_collection(chroma_dataset_name)
vecdb = Chroma(
    client=chroma_client,
    collection_name=chroma_dataset_name,
    embedding_function=embeddings,
)

docs_paths =  [
    '/docs/huggingface_hub/guides/overview',
    '/docs/huggingface_hub/guides/download',
    '/docs/huggingface_hub/guides/upload',
    '/docs/huggingface_hub/guides/hf_file_system',
    '/docs/huggingface_hub/guides/repository',
    '/docs/huggingface_hub/guides/search',
]

base_url = "https://huggingface.co"
filename = "assistantdoc.txt"
root_dir = "./"

content = scrape_all_content(docs_paths, filename)
raw_docs = load_docs(root_dir, filename)
docs = split_docs(raw_docs)

vecdb.add_documents(docs)

