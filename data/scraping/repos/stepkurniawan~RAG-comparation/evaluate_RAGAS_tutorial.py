# !pip install bs4 chromadb tiktoken

#%%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import AzureOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from langchain.chains import LLMMathChain

from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from torch import cuda

import openai
from dotenv import load_dotenv
import os


#%%
load_dotenv()


# openai.api_type = "azure"
# openai.api_base = os.getenv("AZURE_OPENAI_API_ENDPOINT")
# openai.api_version = "2023-07-01-preview"
# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)


#%% load the Wikipedia page and create index

# load the Wikipedia page and create index
loader = WebBaseLoader("https://en.wikipedia.org/wiki/New_York_City") # pip install bs4
index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embed_model,
    # text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # vectorstore_kwargs={ "persist_directory": "/vectorstore"},
).from_loaders([loader]) 


#%% create the QA chain
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    openai_api_key=openai.api_key, 
    model="gpt-3.5-turbo",
    temperature=0.0,
    )

# llm = AzureOpenAI(
#     deployment_name="ChatGPT",
#     openai_api_version = openai.api_version,
#     openai_api_key = openai.api_key,
#     openai_api_base = openai.api_base,
#     openai_api_type = openai.api_type,
#     model_name="ChatGPT",
#     temperature = 0.1,
#     max_tokens = 100,
#     )

#%%

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=index.vectorstore.as_retriever(), 
    return_source_documents=True
)

# testing it out
question = "How did New York City get its name?"
result = qa_chain({"query": question})
result["result"]

# output
# 'New York City got its name in 1664 when it was renamed after the Duke of York, who later became King James II of England. The city was originally called New Amsterdam by Dutch colonists and was renamed New York when it came under British control.'


#%% IMPORT RAGAS METRIC

from ragas.metrics import faithfulness, answer_relevancy, context_relevancy, context_recall
from ragas.langchain import RagasEvaluatorChain

# make eval chains
eval_chains = {
    m.name: RagasEvaluatorChain(metric=m) 
    for m in [faithfulness, answer_relevancy, context_relevancy, context_recall]
}

for name, eval_chain in eval_chains.items():
    score_name = f"{name}_score"
    print(f"{score_name}: {eval_chain(result)[score_name]}")
# %%
