#%%
import os
import openai
import re
import json
import numpy as np
import pandas as pd
import time
openai.api_key = os.environ["OPENAI_API_KEY"]

# openai.Model.list()

#%%
# load data
filedir = ".\\code\\StepGame\\Dataset\\TrainVersion"
qak1_filename = filedir + "/qa1_test.json"
qak2_filename = filedir + "/qa2_test.json"
qak3_filename = filedir + "/qa3_test.json"
qak4_filename = filedir + "/qa4_test.json"
qak5_filename = filedir + "/qa5_test.json"
qak6_filename = filedir + "/qa6_test.json"
qak7_filename = filedir + "/qa7_test.json"
qak8_filename = filedir + "/qa8_test.json"
qak9_filename = filedir + "/qa9_test.json"
qak10_filename = filedir + "/qa10_test.json"
train_filename = filedir + "/train.json"

testk1_data = json.load(open(qak1_filename, 'r'))
testk2_data = json.load(open(qak2_filename, 'r'))
testk3_data = json.load(open(qak3_filename, 'r'))
testk4_data = json.load(open(qak4_filename, 'r'))
testk5_data = json.load(open(qak5_filename, 'r'))
testk6_data = json.load(open(qak6_filename, 'r'))
testk7_data = json.load(open(qak7_filename, 'r'))
testk8_data = json.load(open(qak8_filename, 'r'))
testk9_data = json.load(open(qak9_filename, 'r'))
testk10_data = json.load(open(qak10_filename, 'r'))
train_data = json.load(open(train_filename, 'r'))

#%%
# define the helper function to get the completion
def get_completion(few_shot, ans_format, prompt, model="gpt-3.5-turbo"):
    if few_shot is not None and ans_format is not None:
        messages = [
                    {"role": "user", "content": few_shot},
                    {"role": "assistant","content": ans_format},
                    {"role":"user","content": prompt}
                ]
    else:
        messages = [
                    {"role":"user","content": prompt}
                ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    print(response.choices[0].message["content"])
    return response.choices[0].message["content"]

# corresponding function to extract the answers
def extract_ans(respond):
    pattern = r"\w+:([^;]+);"
    ans = re.findall(pattern, respond)
    return ans

# try to calculate the accuracy
def cal_acc(df):
    df["acc"] = df.apply(lambda x: x["answers"] == x["predicts"], axis=1)
    return df["acc"].mean()

#%%
# build few-shot messages
classes = ["left", "right", "above", "below", "upper-left", 
           "upper-right", "lower-left", "lower-right", "overlap"]

# # one by one 
fewshot_mes = f"""
Analyze the spatial relationship between agents in given story \
that is delimited by triple backticks. Then use the information to answer its following \
question which is delimited by triple dashes.
We assume the North is upper, the South is lower. 12:00 position is upper, 6:00 position is lower. 
Your answer can be only one of the following words or compound modifiers \
delimited by triple quotes: \
\"\"\"{classes}\"\"\"
Answer the question step by step, and you must use the following format:
Answer:<your answer>;
"""


# # original version
# fewshot_mes = f"""
# Analyze the spatial relationship between agents in given story \
# that is delimited by triple backticks. Then use the information to answer its following \
# question which is delimited by triple dashes.
# Each story is independent and not shared. We assume the North is upper, the South is lower. \
#  12:00 position is upper, 6:00 position is lower. 
# For each question, your answer can be only one of the following words or compound modifiers \
# delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# Answer the questions step by step, and you must use the following format:
# text1: <your answer>; text2: <your answer>;...
# Each answer corresponds to its story in order.

# Story text1: ```{' '.join(train_data['0']['story'])}```
# Question text1: ---{train_data['0']['question']}---

# Story text2: ```{' '.join(train_data['15']['story'])}```
# Question text2: ---{train_data['15']['question']}---
# """
# fewshot_ans = f"""text1:{train_data['0']['label']};text2:{train_data['15']['label']};"""


# fewshot_mes = f"""
# Analyze the spatial relationship between agents in given story \
# that is delimited by triple backticks. Then use the information to answer its following \
# question which is delimited by triple dashes.
# Each story is independent and not shared. We assume the North is upper, the South is lower.
# For each question, your answer can be only one of the following words or compound modifiers \
# delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# and you must use the following format:
# text1:<your answer>;text2:<your answer>;...
# Each answer corresponds to its story in order.

# Story text1: ```{' '.join(train_data['0']['story'])}```
# Question text1: ---{train_data['0']['question']}---

# Story text2: ```{' '.join(train_data['15']['story'])}```
# Question text2: ---{train_data['15']['question']}---

# Story text3: ```{' '.join(train_data['12882']['story'])}```
# Question text3: ---{train_data['12882']['question']}---

# Story text4: ```{' '.join(train_data['12907']['story'])}```
# Question text4: ---{train_data['12907']['question']}---

# Story text5: ```{' '.join(train_data['20946']['story'])}```
# Question text5: ---{train_data['20946']['question']}---

# Story text6: ```{' '.join(train_data['20945']['story'])}```
# Question text6: ---{train_data['20945']['question']}---

# Story text7: ```{' '.join(train_data['33252']['story'])}```
# Question text7: ---{train_data['33252']['question']}---

# Story text8: ```{' '.join(train_data['33236']['story'])}```
# Question text8: ---{train_data['33236']['question']}---

# Story text9: ```{' '.join(train_data['46867']['story'])}```
# Question text9: ---{train_data['46867']['question']}---
# """
# # ["left", "right", "above", "below", "upper-left", 
# #     0      15      12882    12907      20946
# # "upper-right", "lower-left", "lower-right", "overlap"]
# #     20945          33252         33236        46867
# fewshot_ans = f"""text1:{train_data['0']['label']};text2:{train_data['15']['label']};text3:{train_data['12882']['label']};\
# text4:{train_data['12907']['label']};text5:{train_data['20946']['label']};text6:{train_data['20945']['label']};\
# text7:{train_data['33252']['label']};text8:{train_data['33236']['label']};text9:{train_data['46867']['label']};"""

print(fewshot_mes)
# print(fewshot_ans)

#%%
# build prompt
import numpy as np
import time

SEED = 0
PNUM = 1
MAXNUM = 40
# K = 2

# for K in range(3,11):
for K in [1]:
    cur_data = eval(f"testk{K}_data")
    for cur_STEP in range(0, min(len(cur_data), MAXNUM), PNUM):
    # for cur_STEP in range(3 * PNUM, min(len(cur_data), MAXNUM), PNUM):
        # prompt = """"""
        prompt = fewshot_mes
        answers = []
        count = 0
        for i in range(cur_STEP, min(cur_STEP + PNUM, min(len(cur_data), MAXNUM))):
            answers.append(cur_data[str(i)]['label'])
#             prompt += f"""
# Story text{i+1}: ```{' '.join(cur_data[str(i)]['story'])}```
# Question text{i+1}: ---{cur_data[str(i)]['question']}---
# """
            prompt += f"""
Story: ```{' '.join(cur_data[str(i)]['story'])}```
Question: ---{cur_data[str(i)]['question']}---
"""
        # prompt = re.sub(pattern, replacement, prompt)
        print(prompt)

        # respond = get_completion(fewshot_mes, fewshot_ans, prompt)
        respond = get_completion(None, None, prompt)
        predicts = extract_ans(respond+("" if respond[-1]==";" else ";"))
        print(predicts)
        print(answers)

        # save
        resultfile = f"{filedir}/../test{K}_{PNUM}.csv"
        # df = pd.DataFrame({"k":K, "answers": answers, "predicts": predicts}, index=None)
        df = pd.DataFrame({"k":[K], "answers": answers, "predicts": [predicts]}, index=None)
        if (not os.path.exists(resultfile)):
            pd_header = pd.DataFrame(columns=["k", "answers", "predicts"])
            pd.DataFrame(pd_header).to_csv(resultfile, index=False, header=True)
        df.to_csv(resultfile, mode='a', index=False, header=False)


        print(cal_acc(df))
        # time.sleep(20)
        if cur_STEP != min(len(cur_data), MAXNUM):
            time.sleep(20)

#%%
# for K in range(2, 11):
for K in [1]:
    # df = pd.read_csv(f"{filedir}/../test{K}_{PNUM}_k9.csv", header=0, index_col=0)
    df = pd.read_csv(f"{filedir}/../test{K}_{PNUM}.csv", header=0, index_col=0)
    print(f'K={K} ACC: ',cal_acc(df))



#############################################################
#%%
# 对换AB主体

instruction = f"""
Analyze the spatial relationship between agents in given story \
that is delimited by triple backticks. Then use the information to answer its following \
question which is delimited by triple dashes.
We assume the North is upper, the South is lower. 12:00 position is upper, 6:00 position is lower. 
Your answer can be only one of the following words or compound modifiers \
delimited by triple quotes: \
\"\"\"{classes}\"\"\"
Answer the question step by step, and you must use the following format:
Answer:<your answer>;
"""
# For each question, 
# Answer the question step by step.
# Answer the question step by step, and briefly explain your reasoning.
# Each story is independent and not shared.
# and you must use the following format:
# text1:<your answer>;text2:<your answer>;...
# Each answer corresponds to its story in order.


# build prompt
for K in [1]:
    cur_data = eval(f"testk{K}_data")
    for cur_STEP in [0]:
        prompt = instruction
        answers = []
        count = 0
        # for i in range(cur_STEP, 3):
        for i in range(cur_STEP, cur_STEP+1):
            answers.append(cur_data[str(i)]['label'])
#             prompt += f"""
# Story text{i+1}: ```{' '.join(cur_data[str(i)]['story'])}```
# Question text{i+1}: ---{cur_data[str(i)]['question']}---
# """
            prompt += f"""
Story: ```{' '.join(cur_data[str(i)]['story'])}```
Question: ---{cur_data[str(i)]['question']}---
"""
        # prompt = re.sub(pattern, replacement, prompt)
        print(prompt)

        respond = get_completion(None, None, prompt)
        # predicts = extract_ans(respond+("" if respond[-1]==";" else ";"))
        # print(predicts)
        print(answers)

        # time.sleep(20)




# """
# 1-0:
# C
# Y
# M



# 2-0:
# Q
# P N
#   K Y

# 1:
# """



















