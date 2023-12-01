from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate

class DocumentSplitter:
    def __init__(self):
        self.document_store = None
    def LoadDocumentStore(self, path_to_document_store): 
        #loads text files and returns the relevant documents -- currently only accepts text files
        loader = DirectoryLoader(path_to_document_store, glob='./*.txt', loader_cls=TextLoader)
        documents = loader.load()
        for i in range(len(documents)): 
            documents[i] = documents[i].replace('\n', ' ')
        self.document_store = documents
    def SplitDocuments(self, chunksize, chunksize_overlap, documents):
        #accepts loaded documents and returns the corresponding parametrically split text data according to the recursive character text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=chunksize_overlap)
        texts = text_splitter.split_documents(documents)
        return texts

class VectorDBCreator:
    def __init__(self, persistence_directory, embeddings, texts): 
        self.persistence_directory = persistence_directory
        self.embeddings = embeddings
        self.split_docs = texts
        self.vectorDB = None
    def ManufacturePersistentDirectory(self):
        self.vectorDB = Chroma.from_documents(documents=self.split_docs, embedding=self.embeddings,persist_directory=self.persistence_directory)
        vectorDB.persist() 
    def LoadVDBDisk(self):
        self.vectorDB = Chroma(persist_directory=self.persistence_directory, embedding_function=self.embeddings)
    def GetVDB(self):
        return self.vectorDB
    def getRetrieverFromVDB(self, search_kwargs={"k": 2}):
        return self.vectorDB.as_retriever(search_type="mmr",search_kwargs=search_kwargs)

class ChainObj: 
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.chain = None
    def ConstructChain(self):
        prompt_template = """
        You are an expert advisor for the Georgia Tech College of Computing who can answer questions based on the following pieces of context. 
        You are responding to a student in the Georgia Tech College of Computing.
        If you do not know the answer based on the context, say you are unsure of the answer. Do not make up an answer.

        {context}

        Question: {question}
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}
        qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=self.retriever, return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
        self.chain = qa_chain
        return qa_chain
    def ProcessResponseSourceLess(self, llm_response):
         return llm_response['result']
    def getResponse(self, query): 
        if self.chain is None: 
            return 
        else:
            return self.chain(query)
    def ProcessResponseSources(self, llm_response): 
        return llm_response['result'], llm_response["source_documents"] 



    
        
        

    
    
