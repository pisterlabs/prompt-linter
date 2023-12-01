from langchain.document_loaders import JSONLoader,TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

med_loader=TextLoader('mashqa_data/sentences.txt')

med_data = med_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
all_splits = text_splitter.split_documents(med_data)


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=hf, persist_directory='./med_db')
vectorstore.persist()
