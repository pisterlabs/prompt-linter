with open("Data\\Api.txt", "r") as file:
    API = file.read()

import openai
from dotenv import load_dotenv

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def QuestionsAnswer(question, chat_log=None):
    chat_log_template = ""

    with open("DataBase\\qna_log.txt", "r") as file:
        chat_log_template = file.read()

    if chat_log is None:
        chat_log = chat_log_template

    prompt = f'{chat_log}Q: {question}\nAns: '
    response = completion.create(model="text-davinci-002", prompt=prompt, temperature=0, max_tokens=100, top_p=1, frequency_penalty=0, presence_penalty=0)
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + f"\nQ: {question}\nAns: {answer}"

    with open("DataBase\\qna_log.txt", "w") as file:
        file.write(chat_log_template_update)

    return answer
'''
while True:
    kk = input("Enter: ")
    if kk == "exit" or kk == "bye":
        exit()
    else:
        print(QuestionsAnswer(kk))
'''