import openai

#SQL产生函数
def sql_generation(
    user_input,
    model="gpt-3.5-turbo-16k",
    temperature=0,
    max_tokens=3000,
    sql_type = 'hive'
    ):

    system_message = f"""
    根据用户的描述写一段{sql_type} SQL代码，仅提供{sql_type} SQL代码：
    """

    messages =  [
        {'role':'system', 'content': system_message},
        {'role':'user', 'content': user_input},
        ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

# 定义合并函数
def merge_textbox(textbox1='', textbox2='', textbox3='', textbox4=''):
    merged_text = f"{textbox1}\n{textbox2}\n{textbox3}\n{textbox4}"
    return merged_text
