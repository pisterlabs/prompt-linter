from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

openAI_chat = ChatOpenAI(temperature=0,request_timeout=100000,max_retries=3)

system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that translates {input_language} to {output_language}.")
human_message_prompt = HumanMessagePromptTemplate.from_template("{input_sentence}")
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt]).format_prompt(input_language="English", output_language="Chinese", input_sentence="I love programming.")

result_prompt = openAI_chat(chat_prompt.to_messages()) 
print(result_prompt.content)