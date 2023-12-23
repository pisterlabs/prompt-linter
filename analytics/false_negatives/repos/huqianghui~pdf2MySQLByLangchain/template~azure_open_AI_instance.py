import os
from langchain import LLMChain
from langchain.prompts import (
    ChatPromptTemplate
)

from langchain.chat_models import AzureChatOpenAI
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://openai-test-fc.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_KEY"] = "XXXX"

chat = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0)
from template.systemTemplate import system_message_prompt
from template.fewshot1 import few_shot_human1, few_shot_ai1
from template.fewshot2 import few_shot_human2, few_shot_ai2
from template.fewshot3 import few_shot_human3, few_shot_ai3
from template.fewshot4 import few_shot_human4, few_shot_ai4
from template.fewshot5 import few_shot_human5, few_shot_ai5

chat_prompt = ChatPromptTemplate.from_messages([
    system_message_prompt,
    few_shot_human1,few_shot_ai1,
    few_shot_human2,few_shot_ai2, 
    few_shot_human3,few_shot_ai3])
# 通过llm和few-shot prompt创建一个chain
chain = LLMChain(llm=chat, prompt=chat_prompt,verbose=True)
