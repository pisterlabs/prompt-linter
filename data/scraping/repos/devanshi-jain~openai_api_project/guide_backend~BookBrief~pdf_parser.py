# Authorization: Bearer OPENAI_API_KEY

# /////////////////////////////////SETUP////////////////////////////////////////
import openai
import os
import re

# import necessary libraries
from dotenv import load_dotenv
from io import StringIO
from pdfminer.layout import LAParams
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator, TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

# from prompt_iteration import get_completion

# Load environment variables from .env file
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) #reads local .env file

# Get value of OPENAI_API_KEY environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI API client
openai.api_key = openai_api_key

# function uses the OpenAI API client to generate text based on a prompt using 
def get_completion(prompt, model="gpt-3.5-turbo"):
    # user prompt (initialized as a list)
    # 2 key-value pairs: role (indicates message is from 'user') and content.
    messages = [{"role": "user", "content": prompt}]
    # response from the api call
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=256,
        temperature=0, # degree of randomness of the model's output
        n = 1 # number of completions to generate for each prompt
    )
    return response.choices[0].message["content"]

def summarize_helper(file, start_page=None, end_page=None):
    output_string = StringIO()
    with file as in_file:
        resource_manager = PDFResourceManager()
        device = TextConverter(resource_manager, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(resource_manager, device)
        parser = PDFParser(in_file)
        document = PDFDocument(parser)
        metadata = document.info[0]
        title = metadata.get('Title', '')
        author = metadata.get('Author', '')
        for page_number, page in enumerate(PDFPage.get_pages(in_file, maxpages=0, caching=True, check_extractable=True), start=1):
            if start_page is not None and page_number < start_page:
                continue
            if end_page is not None and page_number > end_page:
                break
            interpreter.process_page(page)
            content = output_string.getvalue()
            output_string.seek(0)
            output_string.truncate(0)
            prompt = f"""
            Your task is to summarize pages of a book based on the content of the page provided.
            The page text is delimited by triple backticks.
            The summary is intended for someone trying to read the book on their own and need help understanding the content.
            It should be a short paragraph of 4-6 sentences.
            ```{content}```
            """
            response = get_completion(prompt)
            print(f'Page {page_number}:\n{response}')
    device.close()
    response = output_string.getvalue()
    output_string.close()
    return response

with open('/Users/devanshijain/Documents/GitHub/openai_api_project/guide_backend/BookBrief/the-aeneid.pdf', 'rb') as pdf_file:
    pdf_text = summarize_helper(pdf_file,83, 83)
    print(pdf_text)