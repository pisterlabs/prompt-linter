#!/usr/bin/env python3

import json
import openai
import PyPDF2
import sys
import pyperclip
import time

creds = json.load( open("./cred.json"))
openai.organization = creds["openai_org"]
openai.api_key      = creds["openai_api_key"]

system_messages = []
with open("./system.txt") as fd:
    for line in fd:
        system_messages.append({
            "role": "system", "content": line.strip()
        })

## pdfを読み込んで要約させ続ける
result_message = ""
with open(sys.argv[1], 'rb') as file:
    pdf_reader = PyPDF2.PdfReader(file)
    extract    = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text    = extract + "\n"
        text   += page.extract_text().strip()
        message = []
        message += system_messages
        message.append({
            "role": "user", "content": text
        })
        ## go to gpt
        print(message, file=sys.stderr)
        time.sleep(21)
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo"
        ,   messages=message
        )
        print(result["usage"], file=sys.stderr)
        print(result["choices"][0]["finish_reason"])
        result_message = result["choices"][0]["message"]["content"]
        extract        = result_message
        print("------------------------")
        print(result_message)
        print("------------------------")
