import openai
import os
from environs import Env

class OpenAIClient:
    def __init__(self, api_key):
        env = Env()
        env.read_env()
        openai.api_key = env.str('GPT_API_KEY', parse_mode='HTML')
        self.__message = []

    def generate_text(self, prompt):
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5
        )
        return response.choices[0].text

if __name__ == '__main__':
    # Example usage
    api_key = os.environ.get('OPENAI_API_KEY')
    client = OpenAIClient(api_key)
    prompt = input('Enter prompt: ')
    print(client.generate_text(prompt))
