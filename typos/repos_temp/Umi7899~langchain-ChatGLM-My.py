"""现在，请你扮演彩票预测模型。已知根据AI模型在历史开奖数据上的分析，预测得到本周可能的开奖结果为{pred}。

请你根据开奖结果回答用户的问题，用户的问题是{question}
""""""
你现在是一个{role}。这里是一些已知信息：
{related_content}
{background_infomation}
{question_guide}：{input}

{answer_format}
""""""This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
""""""Have a conversation with a human,Analyze the content of the conversation.
You have access to the following tools: """"""Begin!

{chat_history}
Question: {input}
{agent_scratchpad}""""""This is a conversation between a human and a bot:
    
{chat_history}

Write a summary of the conversation for {input}:
""""""Begin!
     
Question: {input}
{agent_scratchpad}"""