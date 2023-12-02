from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import sys
from flask import json
from dotenv import load_dotenv
load_dotenv()

generate_text = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

def text_transform(res):
    return json.dumps({
        'result': res.content,
        'source_documents': []
    })

def build_prompt(messages):
    prompt = []
    for message in reversed(messages):
        previous_message = None
        if message['role'] == 'user':
            previous_message = HumanMessage(content=message['content'])
        elif message['role'] == 'assistant':
            previous_message = AIMessage(content=message['content'])
        if sys.getsizeof(previous_message) + sys.getsizeof(prompt) > 3841:
            break;
        prompt.insert(0, previous_message)
    return prompt

#humanMessage1=HumanMessage(content="what's the capital of france?")
#aiMessage1=AIMessage(content="The capital of France is Paris.")
#humanMessage2=HumanMessage(content="How about Germany?")
#_input_messages = [humanMessage1,aiMessage1,humanMessage2]
#output = generate_text(_input_messages)
#print(output.content)
