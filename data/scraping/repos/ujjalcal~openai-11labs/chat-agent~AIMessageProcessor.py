import json
import os
import boto3
from langchain.chat_models import ChatOpenAI
from chat import Chat
from Agent import Agent
from config import config
from dotenv import load_dotenv

conversation_table_name = config.CONVERSATION_TABLE_NAME
openai_api_key_ssm_parameter_name = config.OPENAI_API_KEY_SSM_PARAMETER_NAME

def lambda_handler(event, context):
    print('lambda_handler - event '+str(event))
    if not is_http_request(event):
        print('lambda_handler - not http request')

        event['body'] = {
            'message': event['inputTranscript'],
            'phoneNumber': event['sessionId']
        }
        event['body'] = json.dumps(event['body'])
        print('lambda_handler - event '+str(event))

    print('lambda_handler - Chat(event)')
    chat = Chat(event)
    print('Lambda_handler - afte - chat = Chat(event) - chat memory '+str(chat.memory))

    set_openai_api_key()
    
    print('Lambda_handler - before get_user_message')
    user_message = get_user_message(event)
    print('Lambda_handler - before is_user_request_to_start_new_conversation')

    #if new convesation rather than a question - end the flow
    if is_user_request_to_start_new_conversation(event):
        print('Lambda_handler - in is_user_request_to_start_new_conversation  - before chat.create_new_chat()')
        chat.create_new_chat()
        response1 = chat.http_response("Hi, I'm Gace, Your Mortgage's digital assistant. What can I help you with today?")
        print('Lambda_handler - in is_user_request_to_start_new_conversation  - response1 '+str(response1))
        return response1
    
    # continue if question
    print('Lambda_handler - before llm chatopenai')
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")


    print('Lambda_handler - initialize langchain_agent with llm and chat.memory - '+str(chat.memory))
    langchain_agent = Agent(llm, chat.memory)
    
    print('Lambda_handler - before langchain_agent.run -with  user_message - '+str(user_message))
    message = langchain_agent.run(input=user_message)
    print('Lambda_handler - after langchain_agent.run')

    if not is_http_request(event):
        lex_response1 = lex_response(event, message)
        print('Lambda_handler - lex_response ')
        return lex_response1
    
    # unreachable
    chat_response = chat.http_response(message)
    print('Lambda_handler - chat_response '+str(chat_response)) 
    return chat_response

def get_session_attributes(intent_request):
    sessionState = intent_request['sessionState']
    if 'sessionAttributes' in sessionState:
        return sessionState['sessionAttributes']
    return {}

def is_user_request_to_start_new_conversation(event):
    print('is_user_request_to_start_new_conversation --> get_user_message '+str(event))
    user_message = get_user_message(event)
    print('is_user_request_to_start_new_conversation --> user_message '+str(user_message))
    outcome = "start a new conversation" in user_message.strip().lower()
    print('is_user_request_to_start_new_conversation --> outcome '+str(outcome))
    return outcome

def is_http_request(event):
    outcome = 'headers' in event
    print('is_http_request - outcome '+str(outcome))
    return outcome

def get_user_message(event):
    print('get_user_message from budy-- event '+str(event))
    body = load_body(event)
    user_message_body = body['message']
    return user_message_body

def load_body(event):
    # if is_http_request(event):
    if True:
        body = json.loads(event['body'])
    else:
        body = json.loads(event['Records'][0]['Sns']['Message'])
    return body

def lex_response(event, message):
    # Return a response to Lex V2
    print('lex_response - start ')
    return {
        'messages': [{
            'contentType': 'PlainText',
            'content': message
        }]
    }
    # return {
    #     'sessionState': {
    #         'sessionAttributes': event['sessionState']['sessionAttributes'],
    #         'dialogAction': {
    #             'type': 'ElicitIntent'
    #         },
    #         'intent': {'name':event['sessionState']['intent']['name'], 'state': 'Fulfilled'}
    #     },
    #     'messages': [{
    #         'contentType': 'PlainText',
    #         'content': message
    #     }]
    # }

def set_openai_api_key():
    print('set_openai_api_key')
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY_SSM_PARAMETER_NAME") #response['Parameter']['Value']
