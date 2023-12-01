import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

# pip install faiss-cpu

url = "https://en.wikipedia.org/wiki/cristiano_Ronalo"

def extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    for script in soup.find_all('script'):
        script.extract()
    return soup.get_text().lower()

print('fetching')
with open('ronaldo.txt', "w") as f:
    f.write(extract_text(url))


print('loader')
loader = TextLoader("ronaldo.txt")
document = loader.load()

print('split')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10, separators=[" ", ",", "\n"])
docs = text_splitter.split_documents(document)

print('embedding')
embedding = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embedding)

llm = HuggingFaceHub(
    repo_id='google/flan-t5-small',
    model_kwargs={
        "temperature": 0.2,
        "max_length": 256
    }
)

print('load qa')
chain = load_qa_chain(llm, chain_type="stuff")

query = "when was cristiano ranaldo born?"
print(query)
docs = db.similarity_search(query)
resp = chain.run(input_documents=docs, question=query)
print(resp)