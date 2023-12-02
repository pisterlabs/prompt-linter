# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/memory/
https://www.langchain.com.cn/modules/memory

默认情况下，Chains和Agents是无状态的，这意味着它们独立地处理每个传入的查询（就像底层的LLMs和聊天模型一样）。
在某些应用程序中（聊天机器人是一个很好的例子），记住以前的交互非常重要，无论是在短期还是长期层面上。
 “记忆”这个概念就是为了实现这一点。
"""
import os
import json
import torch as th
from transformers import (AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          T5Tokenizer, T5ForConditionalGeneration, pipeline)
from langchain.memory import (ChatMessageHistory, ConversationBufferMemory)
from langchain.llms import (OpenAI, HuggingFacePipeline)
from langchain.chains import ConversationChain
from langchain.schema import (messages_from_dict, messages_to_dict)


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 聊天记录
history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_ai_message("whats up?")
history.messages  # [HumanMessage(content='hi!'), AIMessage(content='whats up?')]

# ----------------------------------------------------------------------------------------------------------------
# 缓冲记忆
memory = ConversationBufferMemory()
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")
memory.load_memory_variables({})  # {'history': 'Human: hi!\nAI: whats up?'}
 
memory = ConversationBufferMemory(return_messages=True)
memory.chat_memory.add_user_message("hi!")
memory.chat_memory.add_ai_message("whats up?")
memory.load_memory_variables({})  # {'history': [HumanMessage(content='hi!'), AIMessage(content='whats up?')]}
 
# ----------------------------------------------------------------------------------------------------------------
# 连锁使用
# llm = OpenAI(temperature=0)

checkpoint = "flan-t5-large"  # https://huggingface.co/google/flan-t5-large#usage
tokenizer = T5Tokenizer.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=True
)
pretrained = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=os.path.join(path_model, checkpoint),
    cache_dir=path_model,
    force_download=False,
    local_files_only=False,
    load_in_8bit=False,
    device_map="auto"
    )  # GPU
pipe = pipeline(
    task="text2text-generation",
    tokenizer=tokenizer,
    model=pretrained,
    max_length=100
    )
llm = HuggingFacePipeline(pipeline=pipe)

conversation = ConversationChain(
    llm=llm, 
    memory=ConversationBufferMemory(),
    verbose=True
)

conversation.predict(input="Hi there!")
conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
conversation.predict(input="Tell me about yourself.")
 
# ----------------------------------------------------------------------------------------------------------------
# 保存邮件历史记录
history = ChatMessageHistory()
history.add_user_message("hi!")
history.add_ai_message("whats up?")
 
dicts = messages_to_dict(messages=history.messages)
'''
[{'type': 'human',
  'data': {'content': 'hi!',
   'additional_kwargs': {},
   'type': 'human',
   'example': False,
   'is_chunk': False}},
 {'type': 'ai',
  'data': {'content': 'whats up?',
   'additional_kwargs': {},
   'type': 'ai',
   'example': False,
   'is_chunk': False}}]
'''
new_messages = messages_from_dict(messages=dicts)
'''
[HumanMessage(content='hi!'), AIMessage(content='whats up?')]
'''

# ----------------------------------------------------------------------------------------------------------------
# 对话知识图谱记忆
# https://www.langchain.com.cn/modules/memory/types/kg



