import openai
import os
import json
import requests
from hugchat import hugchat
from hugchat.login import Login
import together
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


from dotenv import load_dotenv
load_dotenv()


TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
AI21_API_KEY = os.getenv('AI21_API_KEY')
ALEPH_API_KEY = os.getenv('ALEPH_API_KEY')
OPEN_ROUTER_API_KEY = os.getenv('OPEN_ROUTER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Huggingface login credentials
HUGGING_EMAIL = os.environ.get("HUGGING_EMAIL")
HUGGING_PASSWORD = os.environ.get("HUGGING_PASSWORD")

MAX_TOKENS = 700


# Log in to huggingface and grant authorization to huggingchat
sign = Login(HUGGING_EMAIL, HUGGING_PASSWORD)
cookie_path_dir = "./cookies"

try:
  cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.

except Exception as e:
  print(e)
  
  # Save cookies to the local directory
  sign.saveCookiesToDir(cookie_path_dir)
  cookies = sign.login()

chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

def hugchat_func(model, params):

    # Create a new conversation
    id = chatbot.new_conversation()
    chatbot.change_conversation(id)

    # get index from chatbot.llms of the model
    index = [i for i, x in enumerate(chatbot.llms) if x == model['api_id']][0]

    print(f"Switching to {index}")

    # set the chatbot to the model
    chatbot.switch_llm(index)

    query_result = chatbot.query(params['text'], temperature=0, max_new_tokens=MAX_TOKENS, stop=params['stop'] if params.get('stop') else None)
    
    return query_result['text']

def together_func(model, params):
    # def format_prompt(prompt, prompt_type):
    #   if prompt_type == "language":
    #       return f"Q: {prompt}\nA: "
    #   if prompt_type == "code":
    #       return f"# {prompt}"
    #   if prompt_type == "chat":
        #   return f"<human>: {prompt}\n<bot>: "
      

    together.api_key = TOGETHER_API_KEY

    # generate response
    response = together.Complete.create(
        model = model['api_id'],
        prompt=f"<human>: {params['text']}\n<bot>:",
        temperature=0,
        max_tokens=MAX_TOKENS,
        stop=["<human>", "<human>:","</s>", "<|end|>", "<|endoftext|>", "<bot>", "```\n```", "\nUser"]
    )


    return response['output']['choices'][0]['text'].rstrip(params['stop'])

def cohere(model, params):
    options = {
        "method": "POST",
        "headers": {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {COHERE_API_KEY}",
        },
        "body": json.dumps({
            "max_tokens": MAX_TOKENS,
            "truncate": "END",
            "return_likelihoods": "NONE",
            "prompt": params['text'],
            "stop_sequences": [params['stop']] if params.get('stop') else [],
            "model": model['api_id'],
            "temperature": 0,
        }),
    }

    response = requests.post("https://api.cohere.ai/v1/generate", headers=options['headers'], data=options['body'])
    json_response = response.json()

    return json_response['generations'][0]['text']

def openai_func(model, params):
    
    openai.api_key = OPENAI_API_KEY

    completion = openai.ChatCompletion.create(
        model=model['api_id'],
        messages=[{"role": "user", "content": params['text']}],
        temperature=0,
        max_tokens=MAX_TOKENS,
        stop=[params['stop']] if params.get('stop') else []
    )
    
    return completion.choices[0].message.content

def ai21(model, params):
    options = {
        "headers": {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {AI21_API_KEY}",
        },
        "body": json.dumps({
            "prompt": params['text'],
            "maxTokens": MAX_TOKENS,
            "temperature": 0,
            "stopSequences": [params['stop']] if params.get('stop') else [],
        }),
    }

    response = requests.post(f"https://api.ai21.com/studio/v1/{model['api_id']}/complete", headers=options['headers'], data=options['body'])
    json_response = response.json()
    return json_response['completions'][0]['data']['text']

def openrouter(model, params):

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "HTTP-Referer": 'https://benchmarks.llmonitor.com', # To identify your app. Can be set to localhost for testing
            "Authorization": "Bearer " + OPEN_ROUTER_API_KEY
        },
        data=json.dumps({
            "model": model['api_id'],
            "temperature": 0,
            "max_tokens": MAX_TOKENS,
            "stop": [params['stop']] if params.get('stop') else [],
            "messages": [ 
                {"role": "user", "content": params['text']}
            ]
        })
    )

    completion = response.json()

    return completion["choices"][0]["message"]["content"]

def anthropic_func(model,params):
    anthropic = Anthropic(
        api_key=ANTHROPIC_API_KEY
    )
    completion = anthropic.completions.create(
        model=model['api_id'],
        temperature=0,
        max_tokens_to_sample=MAX_TOKENS,
        prompt=f"{HUMAN_PROMPT} {params['text']}{AI_PROMPT}",
    )
    return completion.completion

def alephalpha(model, params):
    options = {
        "headers": {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {ALEPH_API_KEY}",
        },
        "body": json.dumps({
            "model": model['api_id'],
            "prompt": params['text'],
            "maximum_tokens": MAX_TOKENS,
            "stop_sequences": [params['stop']] if params.get('stop') else [],
        }),
    }

    response = requests.post("https://api.aleph-alpha.com/complete", headers=options['headers'], data=options['body'])
    json_response = response.json()
    return json_response['completions'][0]['completion']

