"""已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}""""""This is a conversation between a human and a bot:
    
{chat_history}

Write a summary of the conversation for {input}:
""""""Begin!
     
Question: {input}
{agent_scratchpad}"""