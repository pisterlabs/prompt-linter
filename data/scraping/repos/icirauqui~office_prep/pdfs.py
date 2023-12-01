import os
import keys
os.environ["OPENAI_API_KEY"] = keys.open_ai_api_key


from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader


from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

class PDFQA:
    def __init__(self, filename):

        # Method A
        self.loader = PyPDFLoader(filename)
        #self.documents = self.loader.load_and_split()

        # Method B
        self.documents = []
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.add_document(filename)
        print("Num documents: " + str(len(self.documents)))
        #self.documents = self.loader.load()
        
        #self.documents = self.text_splitter.split_documents(self.documents)

        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.documents, 
                                              embedding=self.embeddings, 
                                              persist_directory="db")
        self.vectordb.persist()
        self.memory = ConversationBufferMemory(memory_key="chat_history", 
                                               return_messages=True)

        self.pdf_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo"), 
            self.vectordb.as_retriever(search_kwargs={'k': 1}),
            #self.vectordb.as_retriever(), 
            return_source_documents=True,
            memory=self.memory)
                

    def add_document(self, filename):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(filename)
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            loader = Docx2txtLoader(filename)
        elif filename.endswith('.txt'):
            loader = TextLoader(filename)
        else:
            print("Unsupported file type")
            return

        documents = loader.load()
        self.documents.extend(self.text_splitter.split_documents(documents))


    def qa_basic(self, query, single = True):
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=self.vectordb.as_retriever(search_kwargs={'k': 7}),
            return_source_documents=True
        )

        # we can now execute queries against our Q&A chain
        result = qa_chain({'query': query})
        print(result['result'])

    def qa_history(self, query, chat_history = [], single = True):
        pdf_qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0.8, model_name="gpt-3.5-turbo"), 
            self.vectordb.as_retriever(search_kwargs={'k': 2}),
            #self.vectordb.as_retriever(), 
            return_source_documents=True)
                
        if single:
            print("Q: " + query)
        
        #if history != "":
        #    result = self.pdf_qa({"question": query, "chat_history": history})
        #else:
        #    result = self.pdf_qa({"question": query})
        
        result = pdf_qa({"question": query, "chat_history": chat_history})
        answer = result['answer']
        
        #chain = load_qa_chain(llm=OpenAI())
        #answer = chain.run(input_documents=self.documents, question=query, verbose=True)
        print("A: " + answer)
        chat_history.append((query, answer))
        return answer, chat_history
    
    def run(self):
        chat_history = []
        while True:
            query = input("Q: ")
            if query == "exit" or query == "quit" or query == "q":
                print('Exiting')
                import sys
                sys.exit()
            response, chat_history = self.qa_history(query, chat_history)
            print("")