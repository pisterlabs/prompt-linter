import os
import sys
import json
import openai
# import readline # this module is not meant to work on Windows
# from pyreadline3 import *
# from multiprocessing import Process
import threading

META = ''

def chat(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "system", "content": f"{META}"},
            {"role": "user", "content": f"{question}"}
        ]
    )
    return response['choices'][0]['message']['content']

def readfile(fname):
    try:
        with open(fname, encoding='utf-8') as f:
            return f.read()
    except:
        return None


def writefile(fname, text):
    try:
        with open(fname, 'a+', encoding='utf-8') as f:
            return f.write(text)
    except:
        return None
    
def fchat(fname, question):
    response = chat(question)
    writefile(fname, f'## {question}\n\n{response}')
    return response

def expand(prompt):
    tokens = prompt.split()
    elist = []
    for token in tokens:
        if token.startswith('$'):
            etoken = keys.get(token[1:])
            if etoken is None:
                elist.append(token)
            else:
                elist.append(etoken)
        elif token.startswith('file:'):
            fname = token[5:]
            text = readfile(fname)
            if text is None:
                elist.append(token)
            else:
                elist.append('\n\n'+text)
        else:
            elist.append(token)

    return ' '.join(elist)

openai.api_key = os.getenv("OPENAI_API_KEY")

keys = {
    'mt': '翻譯下列文章',
    'tw': '以 繁體中文 格式輸出',
    'en': 'output in English',
    'jp': 'output in Japanese',
    'md': 'format in Markdown+LaTex, add space before and after $..$'
}

print('Welcome to shortgpt. You may use the following commands')
print('1. quit')
print('2. history')
print('3. shell <command>')
print('4. chat <prompt>')
print('5. fchat <file> <prompt>\n')
print('6. meta <prompt>\n')
print('You may use the following $key for short')
print(json.dumps(keys, indent=2, ensure_ascii=False))

response = None
question = None
commandList = []
while True:
    command = input('\ncommand> ')
    commandList.append(command)

    if command == 'quit':
        break
    elif command.startswith('meta '):
        META = expand(command[5:])
        print("META=", META)
        continue
    elif command.startswith('shell '):
        os.system(command[6:])
        continue
    elif command.startswith('history'):
        for i in range(len(commandList)):
            print(f'{i}:{commandList[i]}')
        continue
    elif command.startswith('chat '):
        prompt = command[5:]
        question = expand(prompt)
        print('========question=======')
        print(question)
        response = chat(question)
        print('========response=======')
        print(response)
        continue
    elif command.startswith('fchat '):
        tokens = command[6:].split()
        fname = tokens[0]
        prompt = ' '.join(tokens[1:])
        question = expand(prompt)
        print('========question=======')
        print(question)
        print('========response=======')
        print(f'Response will write to file:{fname}')
        thread = threading.Thread(target=fchat, args=(fname, question, ))
        thread.start()
        continue
    else:
        print('Command error, try again!')