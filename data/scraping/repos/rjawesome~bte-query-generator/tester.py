from schemas import id_extraction_prompt, json_generation_prompt
import requests
import openai
from dotenv import load_dotenv
import os

load_dotenv()

# specify openai api key as env var OPENAI_API_KEY (or in .env file)
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_ids(question):
    messages_list = [{"role": "system", "content": id_extraction_prompt}, {"role": "user", "content": question}]
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=messages_list)

    ids = []
    active = False
    ind = -1
    for i in chat_completion.choices[0].message.content:
        if i == '[':
            active = True
            ind += 1
            ids.append('')
        elif i == ']':
            active = False
        elif active:
            ids[ind] += i

    return ids    

def resolve_ids(ids):
    prefix_str = ""
    for i in ids:
        url = "https://name-resolution-sri.renci.org/lookup?string=" + i.replace(" ", "%20") + "&offset=0&limit=1"
        res = requests.post(url, data={})
        prefix_str += i + "=" + next(iter(res.json())) + "\n"
    return prefix_str

def get_json(question, resolved_ids):
    messages_list = [{"role": "system", "content": json_generation_prompt}, {"role": "assistant", "content": resolved_ids}, {"role": "user", "content": question}]
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", temperature=0, messages=messages_list)
    
    return chat_completion.choices[0].message.content
    
def question_to_json(question):
    output = 'Server error'
    try: 
        print('Question: ' + question)
        ids = extract_ids(question)
        print("IDs: " + str(ids))
        resolved_ids = resolve_ids(ids)
        print("Resolved IDs: " + resolved_ids)
        output = get_json(question, resolved_ids)
        print("JSON: \n" + output)
    except openai.error.RateLimitError as e:
        output = 'open ai rate limit reached (wait like 1 min) :('
        print('open ai rate limit reached :(')
        
    return output

# using user input
if __name__ == '__main__':
    question = input('Enter a question: ')
    question_to_json(question)
    