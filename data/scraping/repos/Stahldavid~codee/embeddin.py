# %%
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import os

# %%
urls = [
 "https://www.marketsandmarkets.com/Market-Reports/exoskeleton-market-40697797.html",
 "https://www.businesswire.com/news/home/20230517005503/en/Global-Exoskeleton-Market-Research-Report-2022-A-2.5-Billion-Market-by-2028---Analysis-by-Technology-Mobility-Body-Type-End-User---ResearchAndMarkets.com",
 

]


# %%
loader = UnstructuredURLLoader(urls=urls)


# %%
documents = loader.load()



# %%
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# %%
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'

# %%

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, 
    chunk_size=200, 
    chunk_overlap=20
)


# %%
from langchain import HuggingFaceHub

repo_id = "google/flan-t5-xl" # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})

# %%
texts = text_splitter.split_documents(documents)


# %%
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader



embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory='/home/stahlubuntu/chat_docs/chat_fn/chat/camel/')
vectordb.persist()


# %%
for text in texts:

    tokenizer(text.page_content)
    model(text.page_content)
    print("\n\n\n")
    

# %%
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)


# %%
print(texts[1])


