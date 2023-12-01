"""

实体记忆会记住对话中有关特定实体的给定事实。

它提取有关实体的信息（LLM）并随着时间的推移建立有关该实体的知识（llm）。
"""

from langchain.llms import OpenAI
from langchain.memory import ConversationEntityMemory
llm = OpenAI(temperature=0)

memory = ConversationEntityMemory(llm=llm)
_input = {"input": "Deven & Sam are working on a hackathon project"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"output": " That sounds like a great project! What kind of project are they working on?"}
)

memory.load_memory_variables({"input": 'who is Sam'})

memory = ConversationEntityMemory(llm=llm, return_messages=True)
_input = {"input": "Deven & Sam are working on a hackathon project"}
memory.load_memory_variables(_input)
memory.save_context(
    _input,
    {"output": " That sounds like a great project! What kind of project are they working on?"}
)

memory.load_memory_variables({"input": 'who is Sam'})


from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from pydantic import BaseModel
from typing import List, Dict, Any

conversation = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm)
)


conversation.predict(input="Deven & Sam are working on a hackathon project")
