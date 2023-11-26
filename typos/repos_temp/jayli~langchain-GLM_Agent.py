"""你是一个正在跟某个人类对话的机器人.

{chat_history}
人类: {human_input}
机器人:""""""已知信息：
{context}

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请给出你认为最合理的回答。答案请使用中文。 问题是：{question}""""""
你现在是一个{role}。这里是一些已知信息：
{related_content}
{background_infomation}
{question_guide}：{input}

{answer_format}
"""