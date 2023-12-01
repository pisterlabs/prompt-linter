import backoff
import traceback
import sys
import os 
import openai
import tiktoken
import json

from loguru import logger

openai.api_key = os.getenv("OPENAI_API_KEY")


def construct_prompt(cells, prev_messages=None):
    with open('code_explain_sys_msg.txt') as f:
        sys_prompt = ''.join(f.readlines())
    
    if not prev_messages or prev_messages[0].get('role') != 'system':
        if prev_messages is None:
            prev_messages = []
        prev_messages = [{ "role": "system", "content": sys_prompt}] + prev_messages
    
    
    if cells is not None:
        _messages = prev_messages + [{ "role": "user", "content": str(cells)}]
    else:
        _messages = prev_messages
    return _messages

def prompt(
    _messages,
    # model="gpt-4",
    # model='gpt-3.5-turbo-16k-0613',
    model='gpt-3.5-turbo-16k',
    **kwargs
):
    num_tokens = count_tokens_in_prompt_messages(_messages, model_name=model)
    print(f'num_tokens from prompt: {num_tokens}')
    if num_tokens > 16000:
        logger.error('Too many tokens, splitting into multiple prompts')    
        breakpoint()
    
    response = chat_completions_with_backoff(
        model=model,
        messages=_messages,
        temperature=0,
        max_tokens=num_tokens*3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        **kwargs
    )
    
    response_msg = response['choices'][0]['message']
    num_tokens_from_response = count_tokens_in_prompt_messages([response_msg], model_name=model)
    print(f'num_tokens from response: {num_tokens_from_response}')
    return response_msg

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def num_tokens_from_string(string: str, model_name='gpt-3.5-turbo', encoding_name: str=None) -> int:
    """Returns the number of tokens in a text string."""
    if encoding_name:
        encoding = tiktoken.get_encoding(encoding_name)
    elif model_name:
        encoding = tiktoken.encoding_for_model(model_name)
    else:
        raise Exception('Must provide either model_name or encoding_name')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_tokens_in_prompt_messages(messages: list, model_name='gpt-3.5-turbo') -> int:
    """Returns the number of tokens in a list of prompt messages."""
    num_tokens = 0
    for message in messages:
        num_tokens += num_tokens_from_string(message['content'], model_name=model_name)
    return num_tokens


def pprint_msg(assistant_msg, width=100):
    print('='*width)
    print('Assistant message:')
    try:
        prepared_json_format = json.loads(assistant_msg['content'])
    except:
        try:
            prepared_json_format = eval(assistant_msg['content'])
        except:
            prepared_json_format = assistant_msg['content']
            
    print(json.dumps(prepared_json_format, indent=4))
    print('='*width, '\n')
    
# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'a+')
        self.filename = filename
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self
        if self.file.closed:
            self.file = open(self.filename, 'a+')

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()
        

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
        

def get_all_file_with_extension_in_dir_recursively(dir_path, extension):
    import os
    filepaths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(extension):
                filepaths.append(os.path.join(root, file))
    return filepaths