import os
import sys
import json
import openai
# import readline # this module is not meant to work on Windows
# from pyreadline3 import *
import threading

PSTACK = [
    {"role": "system", "content": "You are a chatbot"},
    {"role": "user", "content": "輸出為 繁體中文"},
    {"role": "user", "content": "盡量寫詳細點，每次回答至少輸出 2000 字，以 markdown 格式輸出"},
]

keys = {
    'mt': '翻譯下列文章',
    'tw': '以 繁體中文 格式輸出',
    'en': 'output in English',
    'jp': 'output in Japanese',
    'md': 'format in Markdown+LaTex, add space before and after $..$',
    'mail': '寫一封標題為 $title 的 email，語言為 $lang',
    'book': '請寫一本主題為 $title 的書，用 $lang 書寫，章節盡量細分，每章至少要有 5 個小節，章用 第 x 章，小節前面用 1.1, 1.2 這樣的編號，先寫目錄',
    'lang': '繁體中文', 
    'title': '腦神經心理學與多巴胺',
}

def printStack():
    for i, p in enumerate(PSTACK):
        print(f"{i}:{p['content']}")

def printKeys():
    print(json.dumps(keys, indent=2, ensure_ascii=False))

def chat(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=PSTACK+[{"role": "user", "content": f"{question}"}]
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

def expand1(prompt):
    tokens = prompt.split(' ')
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

def expand(prompt):
    return expand1(expand1(prompt))

def handleCommand(command):
    global commandList, PSTACK
    tokens = command.strip().split(' ')
    op = tokens[0]

    if op == 'quit':
        return
    elif op == 'plist':
        printStack()
    elif op == 'pclear':
        PSTACK = []
        printStack()
    elif op == 'ppush':
        prompt = expand(' '.join(tokens[1:]))
        PSTACK.append({'role':'system', 'content':prompt})
        printStack()
    elif op == 'ppop':
        PSTACK.pop()
        printStack()
    elif op == 'pinsert':
        i = int(tokens[1])
        prompt = expand(' '.join(tokens[2:]))
        PSTACK.insert(i, {'role':'system', 'content':prompt})
        printStack()
    elif op == 'pdelete':
        i = int(tokens[1])
        PSTACK.pop(i)
        printStack()
    elif op == 'klist':
        printKeys()
    elif op == 'kset':
        key = tokens[1]
        value = ' '.join(tokens[2:])
        keys[key] = value
        printKeys()
    elif op == 'shell':
        os.system(' '.join(tokens[1:]))
    elif op == 'history':
        for i in range(len(commandList)):
            print(f'{i}:{commandList[i]}')
    elif op == 'chat':
        prompt = ' '.join(tokens[1:])
        question = expand(prompt)
        print('========question=======')
        print(question)
        response = chat(question)
        print('========response=======')
        print(response)
    elif op == 'fchat':
        fname = tokens[1]
        prompt = ' '.join(tokens[2:])
        question = expand(prompt)
        print('========question=======')
        print(question)
        print('========response=======')
        print(f'Response will write to file:{fname}')
        thread = threading.Thread(target=fchat, args=(fname, question, ))
        thread.start()
    else:
        print('Command error, try again!')


openai.api_key = os.getenv("OPENAI_API_KEY")

print('Welcome to shortgpt. You may use the following commands')
print('* quit')
print('* history')
print('* shell <command>')
print('* chat <prompt>')
print('* fchat <file> <prompt>\n')
print('* plist\n')
print('* pclear\n')
print('* ppush <prompt>\n')
print('* ppop\n')
print('* pinsert <i> <prompt>\n')
print('* pdelete <i>\n')
print('* kset <key> <value>')
print('You may use the following $key for short')
printKeys()

commandList = []
while True:
    command = input('\ncommand> ')
    if command == 'quit': break
    commandList.append(command)
    try:
        handleCommand(command)
    except Exception as err:
        print(f"Error: {err}, {type(err)}")

