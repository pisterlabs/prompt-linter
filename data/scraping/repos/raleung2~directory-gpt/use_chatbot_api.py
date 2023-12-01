'''
Flask api for obtaining vectorstore and interfacing with user regarding directory uploaded
'''
#pylint: disable = line-too-long, no-name-in-module, too-few-public-methods, broad-exception-caught, unused-import

#for documentgpt
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from dotenv import load_dotenv
from pydantic.error_wrappers import ValidationError
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

#for api
from flask import Flask, request
from flask_restful import Resource, Api
from flask import session

#for mongodb
from langchain.memory import MongoDBChatMessageHistory

app = Flask(__name__)
api = Api(app)

class DocumentGPT():
    '''
    document-gpt class for obtaining vectorestore and piping to llm to get completion
    '''
    def __init__(self):
        self.chromadb_folder = "./chromadb/test-db" #folder needs to be specified for api use
        self.embedding_function_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def get_embeddings_from_vectorstore(self):
        '''
        reinitializes vectorstore database into program
        '''
        embedding_function = SentenceTransformerEmbeddings(
            model_name=self.embedding_function_model_name
            )
        database = Chroma(
            persist_directory=self.chromadb_folder,
            embedding_function=embedding_function
            )
        return database

class ChatBot(Resource, DocumentGPT):
    '''
    interfaces with llm with vectorstore and prompt to get completion
    '''
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=os.environ.get('OPENAI_API_KEY')
            )
        #self.llm = GPT4All(
        #model='./model/nous-hermes-13b.ggmlv3.q4_0.bin'
        #max_tokens=2048
        #)

        #connection string for db
        connection_string = os.environ.get("mongo-db-conn-str")

        self.db = MongoDBChatMessageHistory(
            connection_string=connection_string, session_id="test-session"
        )

        self.retriever = self.get_embeddings_from_vectorstore().as_retriever(search_kwargs={"k": 1})
        self.memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=self.db, return_messages=True)
        self.chat = ConversationalRetrievalChain.from_llm(self.llm, retriever=self.retriever, memory=self.memory)

    def post(self):
        '''
        POST request for chatbot
        '''
        data = request.json

        #data validation
        if not data:
            return {'result': 'Invalid request payload'}, 400
        
        prompt = data['message']

        #clears session memory
        if prompt == 'clear':
            self.db.clear()
            return {'result':"Session memory cleared"}, 204

        try:
            completion = self.chat({'question': prompt})

            #add to db
            self.db.add_user_message(prompt)
            self.db.add_ai_message(completion['answer'])

            return {'result': completion['answer']}, 201
        except ValidationError:
            return {'result': ValidationError}, 422
        except Exception as error_message:
            return {'result': str(error_message)}, 500

#add resource to api
api.add_resource(ChatBot, '/api/post')

#run in flask development server
if __name__ == '__main__':
    app.run(debug=True)
