import openai
from main import *
from uuid import uuid4

openai.api_key = os.environ['OPENAI_API_KEY']

params = {'model':'text-davinci-002',
          'max_tokens':1024,
          'temperature':0.7,
          'top_p':1.0,
          'frequency_penalty':0.0,
          'presence_penalty':0.0,
          'stop':None
          }

setting_prompt = """STORYBOT: What sort of setting would you like for your story to take place? If you're not sure, I can brainstorm some ideas."""
antag_prompt = """STORYBOT: Every great story has an antagonist. In can be a character or circumstance. Here are some ideas I came up with."""
start = """STORYBOT: Hi, I’m Storybot, a friendly AI that will help write an amazing story for your little one. Since every great story starts with a main character, can you tell me about your’s? It helps to know their name, age and gender.\nCUSTOMER:"""
finish = """STORYBOT: Is there anything else you would like to add to your story? If not, you can say ALL DONE.\nCUSTOMER: ALL DONE."""

def check_presence(convo, keyword):
    if keyword in convo:
        return True
    else:
        return False
        
def append_prompt(prompt, convo):
    return '\n'.join([convo, prompt])

def append_and_complete(prompt, response, params):
    prompt = append_prompt(prompt, response)
    response = complete(prompt, params)
    return prompt+response.choices[0].text

def append_finish(response):
    if response.choices[0].finish_reason == 'length':
        lines = response.choices[0].text.split('\n')
        if "CUSTOMER:" in lines[-1]:
            lines = lines[:-2]
        elif "STORYBOT" in lines[-1]:
            lines = lines[:-1]
        text = '\n'.join(lines)
    else:
        text = response.choices[0].text
    return '\n'.join([text, finish])


if __name__ == '__main__':
    with open('customer_prompts.txt', 'r') as infile:
        customer_prompts = infile.read().split('\n')

    count = 257
    while count < 400:
        prompt = open_file('storybot_prompt.txt')
        prompt = prompt.replace('<<UUID>>', str(uuid4()))
        prompt = prompt.replace('<<CUSTOMER>>', customer_prompts[count])
        try:
            response = complete(prompt, params, max_retry=1)
            convo = prompt+response.choices[0].text
            output = start + append_finish(response)
            save_file(f'convos/convo_{count}.txt', output)
            print(f'Saved convo_{count}.txt')
            count += 1
        except:
            continue