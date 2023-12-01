from flask import Blueprint, request, jsonify
# from decouple import config
from flask_cors import cross_origin
import json
import cohere
 
co = cohere.Client('nTI4k8KctOt7AiB6JdKIJhnwDUeU1BzauNM6G6Op')

prompt_data = ""
history = []

CHAT_MODEL = 'command'
def talk(prompt):
    global history
    response = co.chat(  
        model='command',  
        message = prompt,  
        temperature=0.3,
        chat_history = history,
        # stream = True,
        prompt_truncation = 'auto',
        citation_quality = 'accurate',
        connectors=[{"id": "web-search"}]
    )
    history.append({'role': 'User', 'message':prompt})
    return response

persona_assignment = '''
    Your name is HRVST. You are an LLM based chatbot that is supposed to answer 
    questions from farmers and hobbyists related to sustainable agricultural practices. 
    You must provide responses to questions asked by the user that include evidence of your claims. 
    Also, identify when the user is wishing to use unsustainable agricultural practices and instead 
    suggest better evidence based options. Most importantly, always explain your reasoning behind a
      decision step-by-step. If a user talks about something they want to do, you must analyze that 
      action for sustainability and possible environmentally harmful consequences.
       '''

talk(persona_assignment)

persona_response = 'what is your name? introduce yourself to the user in a sentence'

talk(persona_response)

main = Blueprint('main', __name__)

@main.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"

@main.route('/add_prompt', methods=['POST'])
def add_prompt():
    global prompt_data
    prompt_data = request.get_json()['text']
    return prompt_data

@main.route('/response')
def response():
    # do the cohere API call here
    global prompt_data
    prompt = prompt_data
    streaming_gens = talk(prompt)

    json_object = {
        "text": streaming_gens.text,
        "documents": streaming_gens.documents
    }

    json_out = json.dumps(json_object)

    return json_out
    # return 