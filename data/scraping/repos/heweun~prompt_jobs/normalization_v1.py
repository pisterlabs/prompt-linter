import pandas as pd
import os
from datetime import datetime
import time, random
from dotenv import load_dotenv
import openai
import re

## 크롤링 병합 결과 출력하기
new = pd.read_csv('./data/0725master.csv')
new = new[new['크롤링날짜'] == '2023-07-24']
new = new[new['크롤링사이트'].isin(['programmers', 'jumpfit', 'wanted'])]

# 인덱스를 없애고 새로운 데이터프레임을 생성
new = new.reset_index(drop=True)
print(new.head())

## chat-GPT활용
load_dotenv()
openai.api_key = os.getenv("openai.api_key")

##프롬프트 입력해서 GPT돌리기
def generate_job_name(Text):
    Job = ['Data Scientist', 'Data Engineer', 'Machine Learning/Deep Learning Researcher', 'Machine Learning/Deep Learning Engineer', 'AI Developer', 'AI Product Service Developer', 'AI Service Planner', 'Prompt Engineer', 'AI Artist', 'AI Researcher', 'AI Engineer']

    delimiter = "####"

    Prompt = f"""
    #Order
    You are a job name classifier.
    Respond directly to the following actions without reiterating the explanation.
    Perform the following actions.
    The text will be delimited with four hashtags, i.e. {delimiter}.
    Step 1: Summarize the following text in 30 words.
    Step 2: Is the summarized information associated with AI? (True or False)
    Step 3: If the answer is true, select only one job name from the following list '{Job}' that describes the summarized information. \
    It must be a job name from the list.\
    If there isn't an appropriate job name, select 'Nan'.\
    If you don't know the answer, just say 'Nan'. DO NOT try to make up an answer.
    
    Respond without reiterating the explanation.
    Don't finish reasoning until you have completed all the steps. 
    You must follow this format:
    
    #Format
    {delimiter} <step 1 reasoning>
    {delimiter} <step 2 reasoning> as a boolean.
    {delimiter} <step 3 reasoning> as a single word only.
    
    Make sure to include {delimiter} to separate every step.
    """

    Assistant = f"""Okay. I understand. I must follow the format.
    I'll respond without reiterating the explanation."""

    messages = [{'role':'system','content':Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'},
                {'role': 'assistant', 'content': Assistant}
    ]
    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo-0613',
        messages = messages,
        temperature = 0
    )

    reply = chat.choices[0].message.content
    print(f'ChatGPT: {reply}','\n')

    try:
        final_response = reply.split(delimiter)[-1].strip()
        ai_response = reply.split(delimiter)[-2].strip()
    except Exception as e:
        final_response = "Error"
        ai_response = "Error"

    print('------Here------'+'\n',ai_response,final_response)
    return (ai_response,final_response)

#직무 선택 반복문
for i in range(len(new)):
    try:
        Text = new.loc[i, '주요업무']+new.loc[i, '자격요건']
        if len(Text) >= 500:
            Text = new.loc[i,'자격요건']
        else:
            Text = Text

        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s]", "", Text).replace('ㆍ', '')

        ai,selected_title = generate_job_name(Text)
        time.sleep(random.uniform(3,5))

        if selected_title:
            new.loc[i, 'AI'] = ai
            new.loc[i, 'category'] = selected_title
        else:
            new.loc[i, 'AI'] = "Error"
            new.loc[i, 'category'] = "Error"
        print("=" * 100)
    except:
        pass

new.to_csv("./data/0725master_v2.csv",index=False)