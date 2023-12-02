# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/model_io/models/chat/
https://www.langchain.com.cn/modules/models/chat/getting_started

本教程涵盖了如何开始使用聊天模型。
通过向聊天模型传递一个或多个消息，您可以获得聊天完成。
响应将是一条消息。LangChain目前支持的消息类型有 HumanMessage、AIMessage、SystemMessage
和 ChatMessage - ChatMessage 接受任意角色参数。
大多数情况下，您只需处理 HumanMessage、AIMessage 和 SystemMessage
"""
import os
import torch as th
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          T5Tokenizer, T5ForConditionalGeneration, pipeline)
from langchain.chat_models import ChatOpenAI
from langchain import (HuggingFaceHub, PromptTemplate, LLMChain)
from langchain.llms import HuggingFacePipeline
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate,
                                    AIMessagePromptTemplate, HumanMessagePromptTemplate)
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
 

print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 1-使用 OpenAI 模型
chat = ChatOpenAI(temperature=0)
chat([HumanMessage(content="Translate this sentence from English to French. I love programming.")])
chat([AIMessage(content="J'aime programmer.", additional_kwargs={})])

messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="I love programming.")
]
chat(messages)

batch_messages = [
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love programming.")
    ],
    [
        SystemMessage(content="You are a helpful assistant that translates English to French."),
        HumanMessage(content="I love artificial intelligence.")
    ],
]
res = chat.generate(batch_messages)


# ----------------------------------------------------------------------------------------------------------------
# 2-使用本地 HuggingFace 模型（根据不同模型的要求定义加载方法）
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


# ----------------------------------------------------------------------------------------------------------------
# 3-集成 PromptTemplate 和 LLMChain
# sys_template = "You are a helpful assistant that translates {input_language} to {output_language}."
# sys_prompt = PromptTemplate(
#     template=sys_template,
#     input_variables=["input_language", "output_language"]
#     )
# sys_msg_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

# hum_template = "{text}"
# hum_prompt = PromptTemplate(
#     template=hum_template,
#     input_variables=["text"]
#     )
# hum_msg_prompt = HumanMessagePromptTemplate(prompt=hum_prompt)

sys_template = "You are a helpful assistant that translates {input_language} to {output_language}."
sys_prompt = SystemMessagePromptTemplate.from_template(sys_template)

hum_template = "{text}"
hum_prompt = HumanMessagePromptTemplate.from_template(hum_template)

chat_prompt = ChatPromptTemplate.from_messages([sys_prompt, hum_prompt])

chat_chain = LLMChain(llm=llm, prompt=chat_prompt)
chat_chain.run(input_language="English", output_language="French", text="I love programming.")
'''
Je l'aime à la programmation.
'''



