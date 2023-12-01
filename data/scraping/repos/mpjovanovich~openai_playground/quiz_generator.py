import argparse
import openai
import os
import PyPDF2

## TODO: Might be worth looking into summarization pipelines
## https://huggingface.co/docs/transformers/tasks/summarization

SYSTEM_PROMPT = '''
The user shall input a document, and the system shall provide quiestions for a college exam based on the provided document. 
The system shall provide a minimum of 10 questions.
Multiple choice questions may or may not include a fill-in-the-blank component.
Questions shall be separated by a blank line.
Questions shall not be prefixed with a number or letter
Responses shall not be prefixed with a number or letter.
The correct response shall be listed first, followed by three incorrect responses. 

Example:
What is a variable?
A placeholder for a value.
A constant value.
The length of a string.
The name of a function.
'''

conversation = [
	## Provide the model with a high level context.
    {"role": "system", "content": SYSTEM_PROMPT},
]

# Note that debugging with launch.json only works if you do the Run > Start Debugging command,
# not if you use the little bug icon.
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--skip_pages', type=str, required=False, default='0')
args = parser.parse_args()

skip_pages = list(map(lambda x: int(x) - 1,args.skip_pages.split(',')))

print('Reading input file...')
text = ''
if args.input_file.endswith('.pdf'):
    with open(args.input_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for i in range(len(pdf_reader.pages)):
            if i in skip_pages:
                continue
            page = pdf_reader.pages[i]
            text += page.extract_text()
else:
    with open(args.input_file, 'r') as file:
        text = file.read()

conversation.append({"role": "user", "content": text})

print('Generating quiz...')
completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    temperature=0.1, #default
    messages=conversation
)
message = completion.choices[0].message.content

print('Writing output file...')
with open(args.output_file, "w") as text_file:
    text_file.write(message)

print('Output file created.')