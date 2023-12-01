import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from getpass import getpass
from uuid import uuid4
from tqdm.auto import tqdm
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent




log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def get_contents(self):
    global embed


OPENAI_API_KEY = "OpenAI_API_Key"
model_name = 'text-embedding-ada-002'


embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)


def get_contents(self):
    pass


YOUR_API_KEY = "Pinecone API KEY"
YOUR_ENV = "asia-southeast1-gcp-free"   


index_name = 'langchain-retrieval-agent'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)


text_field = "text"
index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-4',
    temperature=0.5
)
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)


tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]


agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)


app = Flask(__name__, template_folder='templates')
CORS(app)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        query = data.get('query')
        agent_output = agent(query)
        print(agent_output)  # Add this line
        response = {'response': agent_output['output']}  # Use 'output' here
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        response = agent(message)
        return jsonify({"user_message": message, "chatbot_response": response['output']})  # Use 'action_input' here
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    from threading import Thread
    def run():
        app.run(host='192.168.1.248', port=8080, debug=False)
    t = Thread(target=run)
    t.start()