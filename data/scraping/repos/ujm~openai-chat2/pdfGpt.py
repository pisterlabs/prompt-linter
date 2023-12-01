import openai
import os
import time
from pdfminer.high_level import extract_text

api_key = os.environ['OPENAI_API_KEY']

openai.api_key = api_key

def readPdf(pdf_file):
    #input_pdf = input('pdf: ')
    input_pdf = pdf_file
    text = extract_text(input_pdf)   
    return text

pdf_text = readPdf('bind9-readthedocs-io-en-latest.pdf')

print('[終了するには \"ctrl+c\" と入力してください。]\n\n')
message = {"role":"user", "content":pdf_text}
conversation = [{"role": "system", "content": "システムエンジニア, 日本語, 手順を説明"}]
#conversation.append(message)
#openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)

while(message["content"]!="bye"):
    input_str = ""
    while True:
        if input_str == "":
            line = input('あなた: ')
        else:
            line = input('>>>')
        if line == "":
            break
        input_str += line + "\n"
    # 最後の改行を削除
    input_str = input_str.rstrip('\n')

    message = {"role":"user", "content":input_str};
    conversation.append(message)

    while True:
        try:
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=conversation)
            break
        except:
            time.sleep(5)
            print('処理中...')
            continue

    print(f"アシスタント: {completion.choices[0].message.content}\n")
    conversation.append(completion.choices[0].message)
