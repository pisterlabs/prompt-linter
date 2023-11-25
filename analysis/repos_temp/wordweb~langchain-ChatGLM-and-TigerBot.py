"""已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}""""""
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
{agent_scratchpad}"""