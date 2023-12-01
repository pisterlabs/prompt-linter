from unicodedata import name
import gradio as gr
import pandas as pd
from pathlib import Path
import os
import click
import openai
import sys
from termcolor import colored
api = os.getenv('openapi')   # I hided my api key in here
print(api)
openai.api_key = api

@click.group()
def cli():
    pass

click.group()
@click.command()
def colletdatainteface():
    def collectdata(prompt,instruction,desired_output):
        prompt = prompt.replace('\n', ' ')
        prompt = prompt.rstrip(' ')
        desired_output = desired_output.replace('\n', ' ')
        desired_output = desired_output.rstrip(' ')
        instruction = instruction.replace('\n', ' ')
        instruction = instruction.rstrip(' ')
        prompt = '# '+prompt    .lstrip('\n') + '\n\n# '+instruction + '\n\"\"\"'
        if Path('data.csv').exists():
            df = pd.read_csv('data.csv')
            promps = df['prompt'].values.tolist()
            completion = df['completion'].values.tolist()
            promps.append(prompt)
            completion.append(desired_output)
            df = pd.DataFrame.from_dict({'prompt': promps, 'completion': completion})
            df.to_csv('data.csv', index=False)
        else:
            df = pd.DataFrame.from_dict({'prompt': [prompt], 'completion': [desired_output]})
            df.to_csv('data.csv', index=False)
        return prompt + desired_output +  '\n\"\"\"'

    interface = gr.Interface(collectdata,inputs=['text','text', 'text'],outputs='text')
    interface.launch()

@click.command()
def checkdatacompliance():
    local_file = click.prompt('Please enter the local file path',default='data.csv')
    os.system(f'openai tools fine_tunes.prepare_data -f {local_file}')

@click.command()
def finetunegpt3():
    local_file = click.prompt('Please enter the jsonl file',default='data_prepared.jsonl')
    model = click.prompt('How much do you want to pay',default='text-davinci-002')
    model_name = click.prompt('What do you want to call this model',default='custom_model')
    os.system(f'openai api fine_tunes.create -t {local_file} -m {model} --suffix "{model_name}"')
    
click.command()
def textgeneration():
    model_name = click.prompt('What model do you whant to use',default='text-davinci-002')
    prompt = click.prompt('Specify prompt', default='')
    print('\n\n')
    instruction = click.prompt('Specify instruction', default='')
    api = os.getenv('openapi')   # I hided my api key in here
    openai.api_key = api         # A quik fix is to paste the API here and delete the above line
    """
    Function: text_generation
    -----------------------
    This function takes in a text and returns the generated text.
    param text: The text to be generated.
    type text: str
    return: generated_text: The generated text.
    type generated_text: str
    """

    prompt = '# '+prompt.lstrip('\n') + '\n# '+instruction + '\n\"\"\"'

    response = openai.Completion.create(
    model= model_name,
    prompt= prompt,
    temperature=0.9,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=2,
    presence_penalty=2
    )
    os.system('clear')

    text=colored(response['choices'][0]['text'], 'red', attrs=['reverse', 'blink'])
    print(text)

if __name__ == '__main__':
    cli.add_command(colletdatainteface)
    cli.add_command(checkdatacompliance)
    cli.add_command(finetunegpt3)
    cli.add_command(textgeneration)
    cli()
