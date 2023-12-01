import openai
import click
import os
from dotenv import load_dotenv
from openai.api_resources import engine

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
openai.api_key = OPENAI_KEY

prompt = """Input: Print the current directory
Output: pwd

Input: List files
Output: ls -l

Input: Change directory to /tmp
Output: cd /tmp

Input: Count files
Output: ls -l | wc -l

Input: Replace foo with bar in all python files
Output: sed -i .bak -- 's/foo/bar/g' *.py

Input: Push to master
Output: git push origin master"""

template = """

Input: {}
Output:"""


while True:
    request = input(click.style('nlsh> ', 'blue', bold=True))
    prompt += template.format(request)
    result = openai.Completion.create(
        engine='davinci-codex', prompt=prompt, stop='\n', max_tokens=100, temperature=0.0
    )

    command = result.choices[0]['text']
    prompt += command

    if click.confirm(f'>>> Run: {click.style(command, "blue")}', default=True):
        os.system(command)
