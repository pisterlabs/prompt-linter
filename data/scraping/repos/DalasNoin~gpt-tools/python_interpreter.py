import os
import openai
from io import StringIO
import sys

# compare SergeyKarayev

openai.api_key = os.getenv("OPENAI_API_KEY")
with open('prompt.txt') as f:
    PROMPT = f.read()
STDOUT = sys.stdout

while True:
    # Get user's question
    question = input('Question: ').strip()
    if question == "":
        continue

    # Complete the prompt with user question and call OpenAI
    prompt = f'{PROMPT} {question}\nAnswer:\n```'
    response = openai.Completion.create( model="text-davinci-002", prompt=prompt, temperature=0, max_tokens=512, stop='```', )

    embedding = openai.Embedding.create( response.response )
    
    code = response.choices[0].text.strip()

    # # Print generated code
    # indented_code = '\n'.join([f'\t{line}' for line in code.splitlines()])
    # print(f'[DEBUG] Generated Code:\n{indented_code}')

    # Execute the code!
    sys.stdout = output = StringIO()
    exec(code)
    sys.stdout = STDOUT
    print("Answer: " + output.getvalue())