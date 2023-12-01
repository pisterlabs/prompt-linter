import datetime
import ipynbname
import IPython
import json
import openai
import os

# TODO: Support other LLMs besides chatGPT
# TODO: Test with colab
# TODO: Add some unit tests
# TODO: Can we make it work with star coder?

OPENAPI_API_KEY = "OPENAPI_API_KEY"

_session_started = False

def start_helper(api_key: str = None):
    global _session_started
    
    if api_key is None and OPENAPI_API_KEY in os.environ:
        api_key = os.environ[OPENAPI_API_KEY]
    elif api_key is None:
        print('No API key has been specified.  To generate an OpenAI API key,')
        print('visit this page:')
        print('  https://platform.openai.com/account/api-keys')
        print('')
        print('The API key may be specified by calling this function with the api_key parameter')
        print('or by setting the value of the OPENAPI_API_KEY environment variable before')
        print('starting the Jupyter notebook server.')
        return
    
    openai.api_key = api_key
    print('Usage:')
    print('  ask("some-question") -- submits question to chatGPT and displays response')
    print('  assist("request")    -- submits code request to chatGPT and creates cell with response')
    print('                          click in the cell and press control-enter to execute')
    print('')
    print('Questions, comments and suggestions to matthewcleggphd@gmail.com')
    _session_started = True
    
def _get_notebook():
    IPython.display.display(IPython.display.Javascript('IPython.notebook.save_notebook();'))
    with open(ipynbname.path(), 'r') as file:
        notebook = file.read()
    cell_code = []
    for i, cell in enumerate(json.loads(notebook)['cells']):
        if cell['cell_type'] == 'code':
            cell_code.append(f'### cell {i}\n' + ''.join(cell['source']))
    return '\n'.join(cell_code)

def _create_cell(contents: str):
    escaped_contents = contents.replace('\\', '\\\\').replace('\'', '\\\'').replace('\"', '\\"').replace('\n', '\\n')
    return IPython.display.Javascript("""
        var cell = Jupyter.notebook.insert_cell_below('code')
        cell.set_text("{}")""".format(escaped_contents)
    )

def ask(prompt: str) -> str:
    if not _session_started:
        print('The session has not been started yet.  Please invoke the function start_helper()')
        return
        
    system_prompt = ('You are an expert Python programmer providing assistance to a data scientist. ')
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Here is the code so far:\n" + _get_notebook()},
            {"role": "user", "content": "Please explain the following: " + prompt}
        ]
    )
    code = response["choices"][0]["message"]["content"]
    return IPython.display.Markdown(code)


def assist(prompt: str) -> str:
    if not _session_started:
        print('The session has not been started yet.  Please invoke the function start_helper()')
        return
        
    system_prompt = ('You are an expert Python programmer providing assistance to a data scientist. '
                     +'In your response, please do not include any english text. '
                     +'Your response should consist of Python code only.' 
                     +'If there is any english text in your response, please precede it with an escaped # character.'
                     )
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Here is the code so far:\n" + _get_notebook()},
            {"role": "user", "content": "Please provide Python code to do the following: " + prompt}
        ]
    )
    code = response["choices"][0]["message"]["content"]
    code = code.replace('```python', '')
    code = code.replace('```', '')
    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    code = f'# {formatted_time} {prompt}\n' + code
    return _create_cell(code)
