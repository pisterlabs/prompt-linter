from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/Users/liuziying/working/github/papers/WizardCoder Empowering Code Large Language Models with Evol-Instruct.pdf")
pages = loader.load_and_split()

for page in pages:
    print(page.page_content)

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

print(faiss_index)
