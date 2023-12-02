import pandas as pd
import os
from datetime import datetime
from collections import Counter
import time, random
from dotenv import load_dotenv
import openai
import re

path = 'C:/Users/hoeeun/Desktop/Crawling_master/normalization'
## 크롤링 병합 결과 출력하기
new = pd.read_csv(path + '/data/0731/0731master.csv')
#new = new[new['크롤링날짜'] == '2023-07-31']
new = new[new['crawlingSite'].isin(['점핏', '원티드'])]

# 인덱스를 없애고 새로운 데이터프레임을 생성
new = new.reset_index(drop=True)
print(Counter(new['crawlingDate']))
print(Counter(new['crawlingSite']))

total = len(new)

## chat-GPT활용
load_dotenv()
openai.api_key = os.getenv("openai.api_key")


# assistant있는게 더 나은거 같다.
##프롬프트 입력해서 GPT돌리기
def generate_job_name(Text):
    Job = ['Data Scientist', 'Data Engineer', 'Machine Learning/Deep Learning Researcher',
           'Machine Learning/Deep Learning Engineer', 'AI Developer', 'AI Product Service Developer',
           'AI Service Planner', 'AI prompt Engineer', 'AI Artist']  # 리서처, 엔지니어 -->developer에 추가

    delimiter = "####"

    Prompt = f"""
    #Order
    You are a job name classifier, excelling at precise format adherence.
    Your duty is following the steps to assign a job name for AI job seekers.
    The text will be delimited with four hashtags, i.e. {delimiter}.

    Step 1: Summarize the text in one sentence to guide skills and duties.
    Step 2: Is the summary directly connected to the development, application, and management of machine learning and deep learning technologies? (True or False)
    Step 3: If true, choose one job name exclusively from the provided list '{Job}'.
    You must choose only from the list - nothing else.
    If no suitable job name or unsure, write 'Nan'.
    Avoid providing fabricated responses.

    Absolutely do not finish reasoning until you have completed all the steps.
    You must complete this format:

    #Format
    {delimiter} <step 1 reasoning>\n
    {delimiter} <step 2 reasoning> as a boolean only\n
    {delimiter} <step 3 reasoning> as a single word only\n

    Make sure to separate every step.
    And fill in the entire format.
    Simply show me the results, no further explanations or questions.
    """

    Assistant = f"""I'll respond without reiterating the explanations or questions."""

    messages = [{'role': 'system', 'content': Prompt},
                {'role': 'user', 'content': f'{delimiter}{Text}.'},
                {'role': 'assistant', 'content': Assistant}]

    print(f"messages here:{messages}")

    chat = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        temperature=0
    )

    reply = chat.choices[0].message.content
    print(f'ChatGPT: {reply}', '\n')

    try:
        final_response = reply.split(delimiter)[-1].strip()
        ai_response = reply.split(delimiter)[-2].strip()
        summary = reply.split(delimiter)[-3].strip()
    except Exception as e:
        final_response = "Error"
        ai_response = "Error"
        summary = 'Error'

    print('------Here------' + '\n', f'{i}/{total}\n Summary:{summary}\n AI:{ai_response} Jobname:{final_response}')
    return (summary, ai_response, final_response)


# 직무 선택 반복문
for i in range(total):
    try:
        Text = new.loc[i, 'main'] + new.loc[i, 'require']
        if len(Text) >= 1000:
            Text = new.loc[i, 'require']
        else:
            Text = Text

        Text = re.sub(r'\d+\.', "", Text)
        Text = re.sub(r"[^\w\s~]", "", Text).replace('ㆍ', '')
        Text = re.sub(r'\t', '', Text)

        summary, ai, selected_title = generate_job_name(Text)
        time.sleep(random.uniform(3, 5))

        if selected_title:
            new.loc[i, 'summary'] = summary
            new.loc[i, 'AI'] = ai
            new.loc[i, 'category'] = selected_title
        else:
            new.loc[i, 'summary'] = "Error"
            new.loc[i, 'AI'] = "Error"
            new.loc[i, 'category'] = "Error"
        print("=" * 100)
    except:
        pass

new.to_csv(path + "/data/0731/0731master_v2.csv", index=False)