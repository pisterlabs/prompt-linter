import os
import openai
import wikipedia
import json
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

chat_history = []

def call_openai(message, message_history, functions=[]):
    new_message_history = message_history + [message]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=1,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        messages=new_message_history,
        functions=functions
    )
    assert response['choices']
    response = response['choices'][0]['message']
    message_history.append(message)
    message_history.append(response)
    return response

functions = [{
    "name": "query_wikipedia",
    "description": "query wikipedia to fetch information that helps answer questions about things.",
    "parameters": {
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "person, place, or event the user is asking about"
            }
        },
        "required": ["question", "entity"]
    }
}]

def get_info(entity):
    answer = wikipedia.search(entity)
    names = ', '.join([f'"{s}"' for s in answer])
    print(f'Going to use information about {names} to answer question.')

    content = ''

    for e in answer:
        try:
            page = wikipedia.page(e)
            content += ('\n' + page.content)
        except Exception as ex:
            print(f'Could not find page on {e}')
    # we truncate t he number of characters here so that the number of tokens fit into
    # what the model can handle.
    content = content[:65000]
    return content

def find_question_entity(function_call):
    assert function_call['name'] == 'query_wikipedia'
    arguments = json.loads(function_call['arguments'])
    entity = arguments['entity']
    return entity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    question = args.question

message_history = [
    {'role': 'system', 'content': 'From now on you are a assistant that will help answering question. You should absolutely not use any of your pre-existing knowledge, and only use information in this chat history.'}
]

# first request to openai
user_question = {'role': 'user', 'content': question}

response = call_openai(user_question, message_history, functions=functions)
if 'function_call' not in response:
    print('Could not find entity in question.')
    exit(-1)
    
function_call = response['function_call']

# find the entity the user's question is about
entity = find_question_entity(function_call)

# search wikipedia for information about the entity
print(f'Going to search Wikipedia for {entity}\n')
content = get_info(entity)

# send all information we collected from wikipedia to openai and hope the model can find the answer to the users' question
response = call_openai({
    'role': 'function', 'name': function_call['name'], 'content': content
}, message_history, functions=functions)

print('\n', response['content'])