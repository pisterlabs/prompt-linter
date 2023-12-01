import openai
from openai.error import RateLimitError
import os
from time import sleep
from sys import stderr

openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15' # this may change in the future

def ChatCompletion(deployment: str, prompt: str, retry=3) -> str:
    '''
    Use chat models like completion models.
    '''

    try:
        response = openai.ChatCompletion.create(
            engine=deployment,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content']
    except RateLimitError as e:
        if retry == 0:
            print(f'Fatal: retry times exceeded, "RateLimitError" wrote to completion', file=stderr)
            return 'RateLimitError'
        print(f'RateLimitError, retrying... remaining {retry}', file=stderr)
        sleep(2)
        return ChatCompletion(deployment, prompt, retry - 1)
        
