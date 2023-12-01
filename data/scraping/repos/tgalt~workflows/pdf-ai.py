# Instead of using OCR to extract text, we are using a docx created from a pdf in google drive
# OCR was having too many errors and the docx is much more accurate

import sys
import os
import openai
from docx import Document
from dotenv import load_dotenv

# Load the environment variables from .env file
load_dotenv()

# Get the OpenAI key
openai_key = os.getenv("OPENAI_KEY")

# Set the OpenAI key
openai.api_key = openai_key

def summarize_with_openai(docx_file):
    # Determine the base name of the docx file (without the extension)
    base_name = os.path.splitext(docx_file)[0]

    # Open the docx file
    doc = Document(docx_file)

    # Initialize an empty string to hold the extracted text
    text = ''

    # Loop through each paragraph in the document
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'

    prompt = f"This is the output of an insurance policy pdf from OCR. As such there are errors in the output. The text is:\n\n{text}\n\nPlease put together the best output of the policy limits, deductibles, and coverages.  Please list the named insured but do not list the agent, additional insured's, loss payees etc.  Please list ALL covered vehicles and ALL drivers as well as any business addresses or the insured"

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )

    summary = response.choices[0].message['content']

    # Create the summary file name
    summary_file = base_name + '-ai.txt'

    # Write the summary to a txt file
    with open(summary_file, 'w') as file:
        file.write(summary)

    return summary

# Use the function with command-line arguments
if len(sys.argv) != 2:
    print("Usage: python script.py [input.docx]")
else:
    summary = summarize_with_openai(sys.argv[1])
    print(summary)
