# TalK TO Your PDF

from openai import OpenAI
import PyPDF2
import os

client = OpenAI(
    api_key = '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',  # replace with your own API key
)

# OPenai function
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [    
        # {"role": "system", "content": "You are a language translator, skilled in translate english language to any spoken language"},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )

    return response.choices[0].message.content

# command prompt



# pdf extract function
def extract_text_from_pdf(pdf_filename):
    with open(pdf_filename, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        return text
    
# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Assuming the PDF file name is 'sample.pdf' in the same folder
pdf_file_path = os.path.join(current_directory, 'sample.pdf')

# Extract text from the PDF file
pdf_text = extract_text_from_pdf(pdf_file_path)

text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Give a detail description and outline his qualification .
```{pdf_text}```
"""

response = get_completion(prompt)
print(response)
