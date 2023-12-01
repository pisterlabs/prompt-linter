'''
obtain vectorstore and interface with user regarding directory uploaded
'''
#pylint: disable = unused-import, line-too-long, no-else-break

import os
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from dotenv import load_dotenv

class DocumentGPT():
    '''
    document-gpt class for obtaining vectorestore and piping to llm to get completion
    '''
    def __init__(
        self,
        root_folder:str="./chromadb",
        embedding_function_model_name:str="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ):
        self.root_folder = root_folder #chromadb folder
        self.embedding_function_model_name = embedding_function_model_name

    def select_file_from_root_folder(self):
        '''
        asks user which vectorstore to use
        create vectorstore from create_vectorstore_from_directory.py
        '''
        files = os.listdir(self.root_folder)

        print("Select a vector database file to use:")
        for i, file in enumerate(files):
            print(f"{i+1}. {file}")

        while True:
            choice = input("Enter the number corresponding to the file: ")
            if choice.isdigit() and 1 <= int(choice) <= len(files):
                selected_file = files[int(choice)-1]
                break
            else:
                print("Invalid choice. Please try again.")

        return os.path.join(self.root_folder, selected_file)

    def get_embeddings_from_vectorstore(self):
        '''
        reinitializes vectorstore database into program
        '''
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_function_model_name
            )
        database = Chroma(
            persist_directory=self.select_file_from_root_folder(),
            embedding_function=embedding_function
            )
        return database

    def chatbot(self):
        '''
        interfaces with llm with vectorstore and prompt to get completion
        stores chat history in memory for future recall
        '''
        load_dotenv()
        llm = ChatOpenAI(
          model_name="gpt-3.5-turbo",
          temperature=0,
          openai_api_key=os.environ.get('OPENAI_API_KEY')
          )
        #llm = GPT4All(
          #model='./model/nous-hermes-13b.ggmlv3.q4_0.bin'
          #max_tokens=2048
          #)
        retriever=self.get_embeddings_from_vectorstore().as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chat = ConversationalRetrievalChain.from_llm(llm,retriever=retriever,memory=memory)

        while True:
            prompt = input('Enter your question (-1 to terminate):')

            print(memory)

            if prompt == "-1":
                break

            completion = chat({"question": prompt})
            print('Answer: ', completion['answer'])

instance = DocumentGPT(
  root_folder = './chromadb'
)
instance.chatbot()

#for limited number of documents: set retriever=self.get_embeddings_from_vectorstore().as_retriever(search_kwargs={"k": 1})
#ref: https://github.com/hwchase17/langchain/issues/2255
