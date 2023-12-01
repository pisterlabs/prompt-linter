from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

qa_retrieval_chain = None
llm = None
llm_embeddings = None

def init_llm():
    global llm, llm_embeddings
    llm = OpenAI()
    llm_embeddings = OpenAIEmbeddings()

init_llm()

def extend_knowledge(txt_document_path):
    global qa_retrieval_chain, llm, llm_embeddings
    loader = TextLoader(txt_document_path)    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, llm_embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    qa_retrieval_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever= retriever)

def process_prompt(prompt):
    global qa_retrieval_chain
    result = qa_retrieval_chain.run(prompt)
    return result

# Initialize the language model
