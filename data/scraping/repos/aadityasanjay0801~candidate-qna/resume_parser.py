from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import google_palm
from langchain.chat_models.google_palm import ChatGooglePalm
from operator import itemgetter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from IPython.display import display, Markdown
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
import google.generativeai as palm
import os
import langchain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory




def read_env():
    with open(".env", "r") as f:
        env_variables = f.readlines()

    environment_variables = {}
    for env_variable in env_variables:
        key, value = env_variable.split("=", 1)
        environment_variables[key.replace(" ", "")] = value.replace(" ", "").strip()
    return environment_variables

environment_variables = read_env()

api_key = environment_variables["PaLM_API_KEY"]
os.environ["GOOGLE_API_KEY"] = api_key
palm.configure(api_key=environment_variables["PaLM_API_KEY"])

chat_palm = ChatGooglePalm(
    # cache = True, 
    temperature = 0,)
embeddings_palm = GooglePalmEmbeddings()


def load_document(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return pages

def create_index(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                  chunk_overlap=10,
                                                  length_function = len)
    resume_chunks = text_splitter.transform_documents(pages)
    store = LocalFileStore("./cachce/")
    embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings_palm,
        store,
        namespace = embeddings_palm.model_name
    )

    vectorstore = FAISS.from_documents(resume_chunks, embedding=embeddings_palm)
    retriever = vectorstore.as_retriever()
    return retriever

def get_retriever(query, retriever):
    handler = StdOutCallbackHandler()
    prompt_template_text = """Answer the following question about the Candidate based on his resume, do not answer if it is not in resume,
    answer the question in less than 50 words, prefer bullet points if possible and feasibile.
    Question: {query}
    Answer: ............."""
    prompt_template = PromptTemplate(
        template=prompt_template_text, input_variables=["query"] )
    # chain_type_kwargs = {"prompt": prompt}
    prompt = prompt_template.format(query=query)
    qa_retriever = RetrievalQA.from_chain_type(llm = chat_palm, retriever = retriever,
                                           callbacks = [handler],
                                           return_source_documents = False
                                           
                                        #    chain_type_kwargs = chain_type_kwargs
                                           )
    response = qa_retriever({"query": prompt})
    # response = qa_retriever.run(query)
    return response
    




