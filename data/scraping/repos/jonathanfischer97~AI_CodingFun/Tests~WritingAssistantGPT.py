
import os
import openai
# import deeplake 
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# %%
load_dotenv(os.getcwd()+'/keys.env')
activeloop_token = os.getenv('ACTIVELOOP_TOKEN')
deeplake_username = os.getenv('DEEPLAKE_USERNAME')
openai.api_key = os.environ.get('OPENAI_API_KEY')

pdfdirname = os.getcwd()+'/../LangChainQueryTexts/ThesisPapers'
pdfdirname = os.path.abspath(pdfdirname)

# %%
def load_docs(root_dir):
    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            print(file)
            try:
                loader = PyPDFLoader(os.path.join(
                    dirpath, file))
                docs.extend(loader.load_and_split())
            except Exception as e:
                print(e)
                pass
    return docs


def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def main(root_dir, deep_lake_path):
    docs = load_docs(root_dir)
    texts = split_docs(docs)
    embeddings = OpenAIEmbeddings()
    
    db = DeepLake(dataset_path=deep_lake_path, embedding_function=embeddings)
    
    db.add_documents(texts)

# %% [markdown]
# ## Load, split, and embed text

# %%
docs = load_docs(pdfdirname)
texts = split_docs(docs)

# %%
embeddings = OpenAIEmbeddings()
embeddings

# %% [markdown]
# ## Create Chroma vectorstore for pdfs

# %%
doc_retriever = Chroma.from_documents(texts, embeddings).as_retriever() #initialize vectorstore into retriever

# %%
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor()

# %% [markdown]
# ## Load chat model

# %%
chat = ChatOpenAI(model_name="gpt-4", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

# %%
qa = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=doc_retriever, return_source_documents=True)

# %%
query = "How does membrane localization effect biochemical kinetics?"
result = qa({"query": query})
# qa.run(query)

# %%
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# %%
memory=ConversationBufferMemory()

# %%
convo_qa = ConversationalRetrievalChain.from_llm(chat, doc_retriever,verbose=True, memory=memory)

# %%
convo_qa.predict(input=query)

# %%
from langchain import PromptTemplate, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate


