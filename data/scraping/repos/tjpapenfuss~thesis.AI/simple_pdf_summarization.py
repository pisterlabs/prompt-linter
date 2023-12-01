import os
import PyPDF2
import re
import openai

import config

OPENAI_API_KEY = config.api_key
openai.api_key = OPENAI_API_KEY

# Here I assume you are on a Jupiter Notebook and download the paper directly from the URL
#!curl -o paper.pdf https://arxiv.org/pdf/2301.00810v3.pdf?utm_source=pocket_saves

# Set the string that will contain the summary     
pdf_summary_text = ""
# Open the PDF file
pdf_file_path = "/Users/tannerpapenfuss/thesis.AI/langchain/Ford-Data-Driven.pdf"
# Read the PDF file using PyPDF2
pdf_file = open(pdf_file_path, 'rb')
pdf_reader = PyPDF2.PdfReader(pdf_file)
# Loop through all the pages in the PDF file
for page_num in range(len(pdf_reader.pages)):
    # Extract the text from the page
    page_text = pdf_reader.pages[page_num].extract_text().lower()
    
    response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful research assistant."},
                        {"role": "user", "content": f"Summarize this: {page_text}"},
                            ],
                                )
    page_summary = response["choices"][0]["message"]["content"]
    pdf_summary_text+=page_summary + "\n"
    pdf_summary_file = pdf_file_path.replace(os.path.splitext(pdf_file_path)[1], "_summary.txt")
    with open(pdf_summary_file, "w+") as file:
        file.write(pdf_summary_text)

pdf_file.close()

with open(pdf_summary_file, "r") as file:
    print(file.read())