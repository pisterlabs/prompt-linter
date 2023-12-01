fileopen=open("data\\api.txt")
api=fileopen.read()
fileopen.close()
print(api)


import openai
from dotenv import load_dotenv



openai.api_key=api
load_dotenv()
completion= openai.Completion()

def qna(question,chat_log=None):
    filelog=open("database\qna_log.txt","r")
    chat_log_template= filelog.read()
    filelog.close()

    if chat_log is None:
        chat_log=chat_log_template
    prompt=f'{chat_log} Ques : {question}\n Ans : '
    response=completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    ans=response.choices[0].text.strip()
    chat_log_template_update=chat_log_template+ f"\n Ques : {question} \n Ans : {ans}"
    filelog=open("database\qna_log.txt","w")
    filelog.write(chat_log_template_update)
    filelog.close()
    return ans