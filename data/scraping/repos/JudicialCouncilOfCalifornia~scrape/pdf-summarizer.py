from io import BytesIO
import requests
from PyPDF2 import PdfReader
import pandas as pd
from bardapi import Bard
from numpy import random
from time import sleep
import openai
import os
import re
import sys

filename = sys.argv[1]
openai.api_key = os.environ['_CHATGPT_API_KEY']
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Windows; Windows x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36'}
EOF_MARKER = b'%%EOF'

def reducewords(text, numofwords):
    text = re.sub(' +', ' ', text)
    text = text.replace('\r', '').replace('\n', '')
    return ' '.join(text.split()[:numofwords])

def randomsleep():
    sleeptime = random.uniform(2, 4)
    print("SLEEPING FOR:", sleeptime, "seconds")
    sleep(sleeptime)
    print("SLEEP OVER.")

def fixpdf(pdf_file_on_mem):
    pdf_file_on_mem.write(EOF_MARKER)
    pdf_file_on_mem.seek(0)
    return pdf_file_on_mem

def grabfiletomemory(url):
    response = requests.get(url=url, headers=HEADERS, timeout=120)
    return BytesIO(response.content)

def pdftotext(url):
    pdf_file_on_mem = grabfiletomemory(url)
    pdf_file = PdfReader(pdf_file_on_mem)

    ocr = ''
    for page_number in range(len(pdf_file.pages)):   # use xrange in Py2
        page = pdf_file.pages[page_number]
        page_content = page.extract_text()
        ocr += page_content

    return ocr

def bard(prompt):
    bard = Bard(timeout=30)
    return bard.get_answer(prompt)['content']

def chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a resume, and your task is to rewrite the text into plain English with different paragraphs for each section."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=1025,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content.strip()


pdfs = pd.read_csv(filename)

for i in pdfs.index:

    if pd.isnull(pdfs['summary'][i]):

        try:
            print('PROCESSING: ' + pdfs['url'][i])
            ocr = pdfs['ocr'][i]

            if pd.isnull(ocr):
                print('-- converting PDF to TEXT')
                pdfs.at[i, 'ocr'] = ocr = pdftotext(pdfs['url'][i]).strip()

            # if ocr:
            #     # ocr = reducewords(ocr, 2000)
            #     print('-- asking ChatGPT to summarize')
            #     answer = chatgpt(ocr)
            #     pdfs.at[i, 'summary'] = answer if "Response Error" not in answer else None
            #     print(answer)

        except Exception as err:
            print('FAILED: ' + pdfs['url'][i])
            print(f"Unexpected {err=}, {type(err)=}")
            pass
            # fixed = fixpdf(pdf_file_on_mem)
            # pdfs.at[i, 'ocr'] = pdftotext(fixed)

pdfs.to_csv(filename, index=False)
