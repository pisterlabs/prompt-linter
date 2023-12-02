import os
import sys
import openai

def chatgpt(question, lang, format):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "system", "content": f"Answer in {lang}"},
            {"role": "system", "content": f"Format in {format}"},
            {"role": "user", "content": f"{question}"}
        ]
    )  
    return response['choices'][0]['message']['content']

def translate(text, toLang):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot"},
            {"role": "system", "content": f"Answer in {toLang}"},
            {"role": "system", "content": f"Format in {format}"},
            {"role": "user", "content": f"Translate the following text to {toLang}\n\n{text}"}
        ]
    )
    result = response['choices'][0]['message']['content']
    return result

openai.api_key = os.getenv("OPENAI_API_KEY")

opList = ['quit', 'chatgpt', 'load', 'save', 'shell', 'translate', 'history']

narg = len(sys.argv)
user = sys.argv[1] if narg > 1 else 'user'
lang = sys.argv[2] if narg > 2 else '繁體中文'
format = sys.argv[3] if narg > 3 else 'Markdown+LaTex, add space before and after $..$'
print(f'Welcome {user} to shellgpt. You may use to following commands')
print(f'1. chatgpt <question>\n2. load <file>\n3. save <file>\n4. translate\n5. history\n6. quit\n')

response = None
question = None
commandList = []
while True:
    command = input('\ncommand> ')
    print('')
    commandList.append(command)
    args = command.split()
    if len(args) == 0: continue
    op = args[0]
    tail = ' '.join(args[1:])
    if op not in opList:
        print(f'Operation error, only {opList} allowed!')
        continue
    if op == 'chatgpt':
        question = tail
        response = chatgpt(question, lang, format)
        print(response)
    if op == 'quit': break
    if op == 'shell':
        os.system(tail)
    if op == 'load':
        fname = args[1]
        with open(fname, encoding='utf-8') as fh:
            response = fh.read()
        print(response)
    if op == 'save':
        if response is None:
            print('errro: no response to save!')
        else:
            fname = args[1]
            with open(fname, 'a+', encoding='utf-8') as file:
                file.write(response)
    if op == 'translate':
        toLang = args[1] if len(args) > 1 else lang
        response = translate(response, toLang)
        print(response)
    if op == 'history':
        for i in range(len(commandList)):
            print(f'{i}:{commandList[i]}')
