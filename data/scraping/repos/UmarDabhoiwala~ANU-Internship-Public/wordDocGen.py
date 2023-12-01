import os
import openai
import promptFunc 
import pypandoc
from halo import Halo
import config 

spinner = Halo(text='Loading', spinner='dots')
openai.api_key = config.OPENAI_API_KEY

initPrompt = """
Craft a series of fictitious documents for a role-playing game, focusing on financial companies. 
Ensure that the documents are formatted in a realistic corporate style using Markdown, with appropriate headings, lists, bold text, and italics. 
Each document should be at least 500 words in length. When responding, use the year 2023 if not specified, 
and incorporate obscure pop culture references for names when necessary. Please note that this prompt should not be addressed directly, 
but serve as a guide for subsequent responses.
"""

def WordDocGenerator(sheetype, companyA = "", companyB = "", employeeName = "", employeePosition = "", custom =""):

  prompts = promptFunc.prompts(companyA, companyB, employeeName, employeePosition)
  
  sheetName = (list(prompts.keys()))[sheetype]
  print(sheetName)
  
  spinner.start("Text Generating")
  completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "system", "content": initPrompt},
      {"role": "user", "content": prompts[sheetName]}]
  )
  spinner.succeed("Generation Complete")


  response = completion['choices'][0]["message"]["content"]

  #Writing prompt
  with open(f'files/example.md', 'w') as f:
    f.write(response)


  reference_docx = 'files/reference.docx'
  pandoc_args = [
      '--reference-doc=' + reference_docx,
      ]
  # convert markdown file to docx using Pandoc
  
  
  output = pypandoc.convert_file(f'files/example.md', 'docx', outputfile=f'files/example.docx',extra_args=pandoc_args)
  
  




