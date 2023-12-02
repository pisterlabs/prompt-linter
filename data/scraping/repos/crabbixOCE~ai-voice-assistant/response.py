import openai
from apiKey import openai_key
openai.api_key=openai_key

def generate_response(message):
    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{'role':'system','content':'You are a personal assistant. You give concise answers of no more than 100 words, striving for less if possible. If you need to use more than 100 words, ask for permission to continue.'},
                                                                                    {"role": "user", "content": message}],stream=True)
    for chunk in chat_completion:
        content = chunk["choices"][0].get("delta", {}).get("content")
        stop_reason = chunk['choices'][0]['finish_reason']
        if stop_reason != 'stop' and content is not None:
            yield content
        elif stop_reason == 'stop':
            return "STOP"