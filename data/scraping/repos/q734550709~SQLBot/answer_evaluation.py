import os
import openai
import pandas as pd
import random

# 构建leetcode题目文件的路径
leetcode_path = os.path.join("data", "leetcode_questions.xlsx")

#读取leetcode题目
leetcode_df = pd.read_excel(leetcode_path)

#SQL产生函数
def answer_evaluation(
    user_input,
    all_messages,
    question,
    answer,
    model="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=3000,
    ):

    system_message = f"""
    根据下面的问题（使用<>符号分隔），结合答案（使用####分割符分隔）判断用户的回答是否正确，并给出改进建议
    问题如下:<{question}>
    答案如下:####{answer}####

    请使用中文回复
    """

    history_prompt = []

    for turn in all_messages:
        user_message, bot_message = turn
        history_prompt += [
                {'role': 'user', 'content':user_message},
                {'role': 'assistant', 'content': bot_message}
                         ]
    messages =  [
        {'role':'system', 'content': system_message}] \
        + history_prompt + \
        [{'role':'user', 'content': user_input},
        ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    final_response = response.choices[0].message["content"]

    all_messages+= [(user_input,final_response)]

    return "", all_messages  # 返回最终回复和所有消息

#根据难度随机选择题目
def question_choice(difficulty = '简单'):
    simple_records = leetcode_df[leetcode_df['难度'] == difficulty]
    random_simple_record = simple_records.sample(n=1, random_state=random.seed())

    title = random_simple_record['题目标题'].values[0]
    question_url = random_simple_record['题目地址'].values[0]
    question = random_simple_record['题目'].values[0]
    example = random_simple_record['示例'].values[0]
    answer = random_simple_record['答案'].values[0]
    answer_explain = random_simple_record['可参考解析'].values[0]

    #用于生成答案链接
    title_url = f"""### 本题链接：[{title}]({question_url})"""
    answer_explain = f"""### 答案解析见：[{title}]({answer_explain})"""

    return title_url, question, example, answer, answer_explain
