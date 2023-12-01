# from https://gist.github.com/kennethleungty/7865e0c66c79cc52e9db9aa87dba3d59#file-db_build-py

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def setup_builddb():
    # Load PDF file from data path
    loader = DirectoryLoader('../data/',
                            glob="small_3.txt",
                            loader_cls=TextLoader)
    documents = loader.load()

    # Split text from PDF into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device': 'cuda'})

    # Build and persist FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')



embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cuda'})
vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)

# Create a query
query = "What is granulation tissue?"

# Search the FAISS index
results_with_scores = vectordb.similarity_search_with_score(query)

# Display results
context_list = []
for doc, score in results_with_scores:
    context_list.append(doc.page_content)
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForCausalLM
import torch

contexts = ':'.join(context_list)
print(f"contexts={contexts}")

model_name = "google/flan-t5-large" #AutoModelForSeq2SeqLM
#model_name = "/home/alex/coding/llama/llama-2-7b" #
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,torch_dtype=torch.float16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


inputs = tokenizer(f"Given the context {contexts}: Answer {query}", return_tensors="pt")
outputs = model.generate(**inputs.to('cuda') ,max_new_tokens=200)
print(f"Infer output {tokenizer.batch_decode(outputs, skip_special_tokens=True)}")