from flask_restx import Namespace, Resource
from flask import request
from utils.load_model import load_model
from utils.sessions import load_session, save_session, check_session, set_session
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory, ChatMessageHistory
import time
import copy
from colorama import Fore as c
from auth.authentication import *

answer = Namespace('answer', description='Answer related operations')

memory = None
llm = None

@answer.route('/')
class Answer(Resource):
    @require_api_key
    def post(self):
        '''
        Method to get the answer to a question.
            - Requires an API key in the request header.
            - Requires a question in the request body.
        '''
        start_time = time.time()
        
        global llm
        global template
        global memory

        request_data = request.get_json()
        question = request_data.get("prompt")
        session_id = request_data.get("session_id")

        active_session = check_session()

        if type(session_id) != int:
            return answer.abort(500, 'Session ID must be an integer')

        if str(active_session) != str(session_id):
            # Reset memory and change session
            session = load_session(session_id=session_id)
            memory = ConversationBufferWindowMemory(
                    k=5, 
                    memory_key="chat_history", 
                    chat_memory=session
                    )
            # Change active session
            set_session(session_id=session_id)

        if not llm:
            try:
                llm, template = load_model()
                if session_id is None:
                    session = ChatMessageHistory()
                else:
                    session = load_session(session_id=session_id)
                memory = ConversationBufferWindowMemory(
                    k=5, 
                    memory_key="chat_history", 
                    chat_memory=session
                    )
            except Exception as e:
                error_message = str(e)
                return answer.abort(500, error_message)
            
        if not question:
            return answer.abort(400, "Question not provided in the request")

        # Create prompt from template
        ## input_variables reads the variables from the template
        
        prompt = PromptTemplate(template=template, input_variables=["chat_history", "input"])
        conversation_chain = LLMChain(prompt=prompt, llm=llm, memory=memory, verbose=True)

        # Run the chain and get the stream
        stream = conversation_chain.predict(input=question)
        result = copy.deepcopy(stream)

        if session_id is not None:
            history = memory.buffer_as_messages
            print(f'\n[ {c.YELLOW}SAVING{c.RESET} ] Saving changes of session {session_id}')
            save_session(history=history)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n[ {c.WHITE}MODEL{c.RESET} ] Execution time in seconds: {execution_time}\n")

              
        return {"result": result}

    @require_api_key
    def get(self):
        '''
        Method to get the sessions that are saved in the sessions.json file.
        '''

        sessions = {}

        with open(f"./data/sessions.json", "r") as f:
            data = json.load(f)

        for key in data:
            if key != "Active":
                sessions[key] = data[key]["name"]
        
        return sessions