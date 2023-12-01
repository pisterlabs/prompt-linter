from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

template = """You're a goofy Saatva products expert that is talking to customers.
    You can perform mathematical operations such as averages and sums.
    You should only use the Saatva product database to answer questions, but you
    can extrapolate a little about the products from their names.
    {context}
    Question:{question}"""

prompt = PromptTemplate(template=template,input_variables=["context","question"])

def get_chain(path):
    # Load the documents
    loader = CSVLoader(file_path=path)
    
    data = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(data,embeddings)
    retriever = vectorstore.as_retriever()
    retriever.search_kwargs = {'k':100}

    llm=ChatOpenAI(model_name="gpt-4", temperature=0.3)

    return RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       chain_type_kwargs={"prompt":prompt},
                                       input_key="question")


