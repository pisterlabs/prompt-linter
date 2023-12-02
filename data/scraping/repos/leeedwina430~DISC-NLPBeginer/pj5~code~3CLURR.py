#%%
import os
import openai
import re
import numpy as np
import pandas as pd
import time

openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.Model.list()

#%%
# load data
datadir = '..\\data/CLUTRR'
filedir = f'{datadir}/data_db9b8f04'
testk2_filename = filedir + "/1.2_test.csv"
testk3_filename = filedir + "/1.3_test.csv"
testk4_filename = filedir + "/1.4_test.csv"
testk5_filename = filedir + "/1.5_test.csv"
testk6_filename = filedir + "/1.6_test.csv"
testk7_filename = filedir + "/1.7_test.csv"
testk8_filename = filedir + "/1.8_test.csv"
testk9_filename = filedir + "/1.9_test.csv"
testk10_filename = filedir + "/1.10_test.csv"
train_filename = filedir + "/1.2,1.3,1.4_train.csv"

testk2_data = pd.read_csv(testk2_filename, header=0, index_col=0)   # 38
testk3_data = pd.read_csv(testk3_filename, header=0, index_col=0)
testk4_data = pd.read_csv(testk4_filename, header=0, index_col=0)
testk5_data = pd.read_csv(testk5_filename, header=0, index_col=0)
testk6_data = pd.read_csv(testk6_filename, header=0, index_col=0)
testk7_data = pd.read_csv(testk7_filename, header=0, index_col=0)
testk8_data = pd.read_csv(testk8_filename, header=0, index_col=0)
testk9_data = pd.read_csv(testk9_filename, header=0, index_col=0)
testk10_data = pd.read_csv(testk10_filename, header=0, index_col=0)
train_data = pd.read_csv(train_filename, header=0, index_col=0)

#%%
# Version 1: names fed normally
pattern = r"\[([^]]+)\]"
replacement = r"\1"
# replacement = r"'\1'" 

train_data["story"] = train_data["story"].apply(lambda x: re.sub(pattern, replacement, x))
train_data["query"] = train_data["query"].apply(lambda x: re.sub(r"'", r"", x))
# train_data["query"] = train_data["query"].apply(lambda x: re.sub(r"'(.*?)'", r'[\1]', x))

for cur_data in [train_data, testk2_data, testk3_data, testk4_data, testk5_data, 
                testk6_data, testk7_data, testk8_data, testk9_data, testk10_data]:
    cur_data.loc[:,'story'] = cur_data['story'].apply(lambda x: re.sub(pattern, replacement, x))
    cur_data.loc[:,'query'] = cur_data['query'].apply(lambda x: re.sub(r"'", r"", x))
    # cur_data.loc[:,'query'] = cur_data['query'].apply(lambda x: re.sub(r"'(.*?)'", r'[\1]', x))


#%%
# Version 2: names are replaced
pattern = r"\[([^]]+)\]"
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

np.random.seed(0)

def replaceName(row):
    new_row = row.copy()
    sentence = row["story"]
    query = row['query']

    cur_dic = {}
    brackets = re.findall(r"\[.*?\]", sentence)
    new_sentence = sentence
    cur_ALPHA = np.random.permutation(list(ALPHA))
    for i,bracket in enumerate(set(brackets)):
        # cur_dic[bracket[1:-1]] = 'person '+cur_ALPHA[i]
        cur_dic[bracket[1:-1]] = cur_ALPHA[i]
        new_sentence = re.sub(re.escape(bracket), cur_dic[bracket[1:-1]], new_sentence)
    
    brackets = re.findall(r"'.*?'", query)
    new_query = query
    for bracket in set(brackets):
        new_query = re.sub(re.escape(bracket), cur_dic[bracket[1:-1]], new_query)

    new_row.loc['story'] = new_sentence
    new_row.loc['query'] = new_query
    return new_row

for cur_data in [train_data, testk2_data, testk3_data, testk4_data, testk5_data, 
                testk6_data, testk7_data, testk8_data, testk9_data, testk10_data]:
    cur_data.loc[:, ["story", 'query']] = cur_data[["story", 'query']].apply(
                lambda x: replaceName(x), axis=1)

# train_data.loc[:, ["story", 'query']] = train_data[["story", 'query']].apply(lambda x: replaceName(x), axis=1)

#%%
# define the helper function to get the completion
def get_completion(few_shot, ans_format, prompt, model="gpt-3.5-turbo"):
    messages = [
                {"role": "user", "content": few_shot},
                {"role": "assistant","content": ans_format},
                {"role":"user","content": prompt}
            ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    print(response.choices[0].message["content"])  # TODO: 改成查看token数的
    # time.sleep(20)
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
classes = ', '.join(list(train_data.target.value_counts().keys()))

# for name in train_data.task_name.value_counts().keys():

# Say \"No\" if you cannot answer the question.
# TODO: 可以试一下有语法错误是否会影响准确率。guess：会
# fewshot_mes = f"""
# Analyze the kinship between agents through information \
# that is delimited by triple backticks. Then use the information to answer: How is \
# the first agent related to the second agent, which are delimited by triple dashes?
# For each question, your answer can be only one of the following words or compound modifiers\
# delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# and you must use the following format:
# text1:<your answer>;text2:<your answer>;...

# Each information is independent and not shared, and each answer corresponds to its \
# information in order.

# Information text1: ```{train_data.iloc[0].story}```
# Query text1: ---{train_data.iloc[0].query[1:-1]}---

# Information text2: ```{train_data.iloc[-1].story}```
# Query text2: ---{train_data.iloc[-1].query[1:-1]}---

# Information text3: ```{train_data.iloc[-1].story}```
# Question text3: ---How is {train_data.iloc[-1].query[1:-1].split(', ')[1]} related to {train_data.iloc[-1].query[1:-1].split(',')[0]}?---
# fewshot_ans = f"""text1:{train_data.iloc[0].target};text2:{train_data.iloc[5064].target};text3:{train_data.iloc[-1].target};"""

# """


fewshot_mes = f"""
Analyze the kinship between family members in given information \
that is delimited by triple backticks. Then use the information to answer the following \
question which is delimited by triple dashes.
Each information is independent and not shared. Kinship will not exceed three generations.
For each question, your answer can be only one of the following words or compound modifiers \
delimited by triple quotes: \
\"\"\"{classes}\"\"\"
You must use the following format:
text1:<your answer>;text2:<your answer>;...
Each answer corresponds to its information in order.

Information text1: ```{train_data.iloc[0].story}```
Question text1: ---How is {train_data.iloc[0].query[1:-1].split(', ')[1]} related to {train_data.iloc[0].query[1:-1].split(',')[0]}?---

Information text2: ```{train_data.iloc[5064].story}```
Question text2: ---How is {train_data.iloc[5064].query[1:-1].split(', ')[1]} related to {train_data.iloc[5064].query[1:-1].split(',')[0]}?---

Information text3: ```{testk4_data.iloc[-1].story}```
Question text3: ---How is {testk4_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk4_data.iloc[-1].query[1:-1].split(',')[0]}?---

Information text4: ```{testk5_data.iloc[-1].story}```
Question text4: ---How is {testk5_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk5_data.iloc[-1].query[1:-1].split(',')[0]}?---

Information text5: ```{testk6_data.iloc[-1].story}```
Question text5: ---How is {testk6_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk6_data.iloc[-1].query[1:-1].split(',')[0]}?---

Information text6: ```{testk7_data.iloc[-1].story}```
Question text6: ---How is {testk7_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk7_data.iloc[-1].query[1:-1].split(',')[0]}?---

Information text7: ```{testk8_data.iloc[-1].story}```
Question text7: ---How is {testk8_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk8_data.iloc[-1].query[1:-1].split(',')[0]}?---

Information text8: ```{testk9_data.iloc[-1].story}```
Question text8: ---How is {testk9_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk9_data.iloc[-1].query[1:-1].split(',')[0]}?---
"""
# NOTE: 全给他效果不一定就是好的。也可能忘记了前面的prompt
fewshot_ans = f"""text1:{train_data.iloc[0].target};text2:{train_data.iloc[5064].target};text3:{testk4_data.iloc[-1].target};\
text4:{testk5_data.iloc[-1].target};text5:{testk6_data.iloc[-1].target};text6:{testk7_data.iloc[-1].target};\
text7:{testk8_data.iloc[-1].target};text8:{testk9_data.iloc[-1].target};"""

# text9:{testk10_data.iloc[-1].target};


# # Method 2
# query1 = re.sub(r"'(.*?)'", r'[\1]', train_data.iloc[0].query[1:-1])
# query2 = re.sub(r"'(.*?)'", r'[\1]', train_data.iloc[1].query[1:-1])
# fewshot_mes = f"""
# Analyze the kinship between family members in given information \
# that is delimited by triple backticks. Then use the information to answer the following \
# question which is delimited by triple dashes.
# Each information is independent and not shared.
# For each question, your answer can be only one of the following words or compound modifiers \
# delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# and you must use the following format:
# text1:<your answer>;text2:<your answer>;...
# Each answer corresponds to its information in order.

# Information text1: ```{train_data.iloc[0].story}```
# Question text1: ---How is {query1.split(', ')[1]} related to {query1.split(',')[0]}?---

# Information text2: ```{train_data.iloc[1].story}```
# Question text2: ---How is {query2.split(', ')[1]} related to {query2.split(',')[0]}?---
# """

# fewshot_ans = f"""text1:{train_data.iloc[0].target};text2:{train_data.iloc[1].target};"""



# fewshot_mes = f"""
# Your task is to perform the following actions: 
# 1 - Restate the question with your understanding.
# 2 - Use the information delimited by triple backticks to answer the question \
# which is delimited by triple dashes. Your answer can be only one of the following \
# words or compound modifiers delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# 3 - Explain your reasoning step by step.

# Information: ```{train_data.iloc[0].story}```
# Question: ---How is {train_data.iloc[0].query[1:-1].split(', ')[1]} related to {train_data.iloc[0].query[1:-1].split(',')[0]}?---
# """

# fewshot_ans = f"""text1:{train_data.iloc[0].target};reasoning:(Dorothy, Michael)=brother, (Donald, Michael)=father;"""

# # Method 1
# fewshot_mes = f"""
# Analyze the kinship between family members in given information \
# that is delimited by triple backticks. Then use the information to answer the following \
# question which is delimited by triple dashes.
# Each information is independent and not shared.
# For each question, your answer can be only one of the following words or compound modifiers \
# delimited by triple quotes: \
# \"\"\"{classes}\"\"\"
# and you must use the following format:
# text1:<your answer>;text2:<your answer>;...
# Each answer corresponds to its information in order.

# Information text1: ```{train_data.iloc[0].story}```
# Question text1: ---How is {train_data.iloc[0].query[1:-1].split(', ')[1]} related to {train_data.iloc[0].query[1:-1].split(',')[0]}?---

# Information text2: ```{train_data.iloc[5064].story}```
# Question text2: ---How is {train_data.iloc[5064].query[1:-1].split(', ')[1]} related to {train_data.iloc[5064].query[1:-1].split(',')[0]}?---
# """
# Information text2: ```{train_data.iloc[5064].story}```
# Question text2: ---How is {train_data.iloc[5064].query[1:-1].split(', ')[1]} related to {train_data.iloc[5064].query[1:-1].split(',')[0]}?---

# Information text2: ```{train_data.iloc[1].story}```
# Question text2: ---How is {train_data.iloc[1].query[1:-1].split(', ')[1]} related to {train_data.iloc[1].query[1:-1].split(',')[0]}?---

# Information text9: ```{testk10_data.iloc[-1].story}```
# Question text9: ---How is {testk10_data.iloc[-1].query[1:-1].split(', ')[1]} related to {testk10_data.iloc[-1].query[1:-1].split(',')[0]}?---

# fewshot_ans = f"""text1:{train_data.iloc[0].target};text2:{train_data.iloc[5064].target};"""

print(fewshot_mes)
print(fewshot_ans)

#%%
# 纵向拼接版
# build prompt
import numpy as np
import time

SEED = 0
PNUM = 8
MAXNUM = 40
# K = 2

for K in range(5,11):
# for K in [4]:
    cur_data = eval(f"testk{K}_data")
    for cur_STEP in range(0, min(len(cur_data), MAXNUM), PNUM):
    # for cur_STEP in range(1 * PNUM, min(len(cur_data), MAXNUM), PNUM):
        prompt = """"""
        answers = []
        count = 0
        for i in range(cur_STEP, min(cur_STEP + PNUM, min(len(cur_data), MAXNUM))):
            answers.append(cur_data.iloc[i].target)
            # query = re.sub(r"'", r"", cur_data.iloc[i].query[1:-1]).split(', ')
            # query = re.sub(r"'(.*?)'", r'[\1]', cur_data.iloc[i].query[1:-1]).split(', ')
            query = cur_data.iloc[i].query[1:-1].split(', ')
            prompt += f"""
Information text{i+1}: ```{cur_data.iloc[i].story}```
Question text{i+1}: ---How is {query[1]} related to {query[0]}?---
"""
        # prompt = re.sub(pattern, replacement, prompt)
        print(prompt)

        respond = get_completion(fewshot_mes, fewshot_ans, prompt)
        predicts = extract_ans(respond+("" if respond[-1]==";" else ";"))
        print(predicts)
        print(answers)

        # save
        resultfile = f"{datadir}/train23456789_test{K}_origin_{PNUM}.csv"
        df = pd.DataFrame({"k":len(answers), "answers": answers, "predicts": predicts}, index=None)
        if (not os.path.exists(resultfile)):
            pd_header = pd.DataFrame(columns=["k", "answers", "predicts"])
            pd.DataFrame(pd_header).to_csv(resultfile, index=False, header=True)
        df.to_csv(resultfile, mode='a', index=False, header=False)


        print(cal_acc(df))
        time.sleep(20)

#%%
for K in range(2, 11):
# for K in [2]:
    # df = pd.read_csv(f"{datadir}/train234_test{K}_origin_{PNUM}_23.csv", header=0, index_col=0)
    # df = pd.read_csv(f"{datadir}/train234_test{K}_replace_{PNUM}_method1.csv", header=0, index_col=0)
    df = pd.read_csv(f"{datadir}/train23456789_test{K}_origin_{PNUM}.csv", header=0, index_col=0)
    print(f'K={K} ACC: ',cal_acc(df))


#%%
# 纵向拼接版
STEP = 4    # 4

prompt = """"""
answers = []
count = 0
cur_data = eval(f"testk{K}_data")
for i in range(PNUM * STEP, min(PNUM * (STEP+1), len(cur_data))):
    print("i:", i)
    answers.append(cur_data.iloc[i].target)
    query = re.sub(r"'", r"", cur_data.iloc[i].query[1:-1]).split(', ')
    prompt += f"""
Information text{i+1}: ```{cur_data.iloc[i].clean_story}```
Question text{i+1}: ---How is {query[1]} related to {query[0]}?---
"""
prompt = re.sub(pattern, replacement, prompt)

print(prompt)
print(answers)

#%%
# respond = get_completion(fewshot_mes, fewshot_ans, prompt)
# predicts = extract_ans(respond+("" if respond[-1]==";" else ";"))
# print(predicts)
# print(answers)

# df = pd.DataFrame({"k":[K] * min(PNUM, len(cur_data)-PNUM*STEP), "answers": answers, "predicts": predicts}, index=None)
df = pd.DataFrame({"k":[K] * PNUM, "answers": answers, "predicts": predicts}, index=None)
# df.to_csv(f"{datadir}/{time.ctime()[-13:-11]+time.ctime()[-10:-8]+time.ctime()[-7:-5]}_k23_{PNUM}_{OFFSET}.csv")
# df.to_csv(f"{datadir}/train23_test{K}_origin_{PNUM}_{STEP}.csv")
df.to_csv(f"{datadir}/train23_test{K}_origin_{PNUM}.csv")

# df = pd.read_csv(f"{datadir}/train23_test{K}_origin_{PNUM}_{STEP}.csv", header=0, index_col=0)
df = pd.read_csv(f"{datadir}/train23_test{K}_origin_{PNUM}.csv", header=0, index_col=0)


print(cal_acc(df))

#%%
# df = pd.read_csv(f"{datadir}/train23_test{K}_origin_{PNUM}_{STEP}.csv", header=0, index_col=0)
df = pd.read_csv(f"{datadir}/train23_test{K}_origin_{PNUM}.csv", header=0, index_col=0)

# try to calculate acc using code
def cal_acc(df):
    df["acc"] = df.apply(lambda x: x["answers"] == x["predicts"], axis=1)
    return df["acc"].mean()

print(cal_acc(df))

# %%



#%%
# # 横向拼接版
# # build prompt
# import numpy as np
# import time

# SEED = 0
# PNUM = 2    # 3 开始生成格式非想要 (27 questions)
# OFFSET = 20

# np.random.seed(SEED)
# prompt = """"""
# answers = []
# count = 0
# for k in range(2, 11):
#     cur_data = eval(f"testk{k}_data")
#     for i in range(PNUM):
#         count += 1
#         # i = np.random.randint(0, len(cur_data)) + OFFSET
#         i += OFFSET
#         print("i:", i)
#         answers.append(cur_data.iloc[i].target)
#         query = re.sub(r"'", r"", cur_data.iloc[i].query[1:-1]).split(', ')
#         prompt += f"""
# Information text{count}: ```{cur_data.iloc[i].story}```
# Question text{count}: ---How is {query[1]} related to {query[0]}?---
# """
# prompt = re.sub(pattern, replacement, prompt)

# print(prompt)
# print(answers)

#%%
# to change the agents
import re

sentence = "This is an [example] sentence [with brackets]. [example]"
replacement_word = ['A','B','C']

# Task 1: Find all the brackets
brackets = re.findall(r"\[.*?\]", sentence)

print(brackets)

# Task 2: Replace each pair of brackets and the words inside them with a different word
new_sentence = sentence
for i, bracket in enumerate(set(brackets)):
    new_sentence = re.sub(re.escape(bracket), replacement_word[i], new_sentence)

print(new_sentence)
# Output: "This is an replacement sentence replacement."

#%%
# replace names with placeholders
STEP = 4    # 4

prompt = """"""
answers = []
count = 0
cur_data = eval(f"testk{K}_data")
for i in range(PNUM * STEP, min(PNUM * (STEP+1), len(cur_data))):
    print("i:", i)
    answers.append(cur_data.iloc[i].target)
    query = re.sub(r"'", r"", cur_data.iloc[i].query[1:-1]).split(', ')
    prompt += f"""
Information text{i+1}: ```{cur_data.iloc[i].clean_story}```
Question text{i+1}: ---How is {query[1]} related to {query[0]}?---
"""
prompt = re.sub(pattern, replacement, prompt)

print(prompt)
print(answers)


#%%
# draw the tradeoff plot
import matplotlib.pyplot as plt

ks = np.arange(2,11)
accs8 = [0.632,0.225,0.5,0.125,0.3,0.15,0.175,0.125,0.325]
# accs16 = [0.259,0.225,0.35,0.1,0.275,0.05,0.15,0.2,0.175]
accs22 = [0.579,0.275,0.325,0.15,0.275,0.275,0.25,0.225,0.2]
accs23 = [0.552,0.3,0.175,0.175,0.25,0.25,0.275,0.225,0.2]
accs234 = [0.474,0.325,0.225,0.125,0.175,0.25,0.4,0.25,0.2]
accs23456789 = [0.421,0.325,0.325,0.275,0.275,0.175,0.275,0.2,0.175]


accsbracket = [0.5,0.2,0.275,0.175,0.25,0.3,0.325,0.225,0.2]
accsABCD = [0.263,0.1,0.3,0.175,0.25,0.2,0.175,0.225,0.1]

# plt.plot(ks, accs8)
# plt.plot(ks, accs16)
# plt.plot(ks, accs22)
# plt.plot(ks, accs23)
# plt.plot(ks, accs234)
# plt.plot(ks, accs23456789)
plt.plot(ks, accs23)
plt.plot(ks, accsbracket)
plt.plot(ks, accsABCD)


# plt.title("Accuracy-Compositionality Tradeoff Plot")
# plt.title("#few-shot Tradeoff Plot")
plt.title("Names Tradeoff Plot")
plt.xlabel("Compositionality K")
plt.ylabel("Accuracy")
# plt.legend(['train22','train23','train234','train23456789'])
plt.legend(['original','bracketed','ABCD'])
