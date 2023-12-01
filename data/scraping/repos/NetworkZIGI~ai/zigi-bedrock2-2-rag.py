__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from utils import bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  
os.environ["AWS_PROFILE"] = "zigi-bedrock"

boto3_bedrock = bedrock.get_bedrock_client(
    # IAM User에 Bedrock에 대한 권한이 없이 Role을 Assume하는 경우
    # assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None), 
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)


# - create the Anthropic Model
llm = Bedrock(
    model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample": 1000}
)
bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-text-v1")

docsearch = Chroma(persist_directory="./zigi_chromadb", embedding_function=bedrock_embeddings)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer in Korean:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type='stuff', # end: stuff , kor : map_reduce
                                 retriever=docsearch.as_retriever(search_kwargs={'k': 3 }),#, 'score_threshold':3.4375}), ##score_threshold
                                 return_source_documents=True,
                                 chain_type_kwargs=chain_type_kwargs)


query = "고재성이 운영하는 커뮤니티와 블로그를 알려줘"
result = qa({"query": query})
answer = result['result']
print(f"질문 : {query}")
print(f"답변 : {answer}")








#########
'''
from urllib.request import urlretrieve

os.makedirs("data", exist_ok=True)
files = [
    "https://www.irs.gov/pub/irs-pdf/p1544.pdf",
    "https://www.irs.gov/pub/irs-pdf/p15.pdf",
    "https://www.irs.gov/pub/irs-pdf/p1212.pdf",
]
for url in files:
    file_path = os.path.join("data", url.rpartition("/")[2])
    urlretrieve(url, file_path)
    
    
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("./data/")

documents = loader.load()
# - in our testing Character split works better with this PDF data set
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100,
)
docs = text_splitter.split_documents(documents)

sample_embedding = np.array(bedrock_embeddings.embed_query(docs[0].page_content))


query = "Is it possible that I get sentenced to jail due to failure in filings?"

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)
query = "Is it possible that I get sentenced to jail due to failure in filings?"
result = qa({"query": query})
print(result["result"])

'''
