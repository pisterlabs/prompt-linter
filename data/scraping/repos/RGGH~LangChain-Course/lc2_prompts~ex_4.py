from langchain.llms import OpenAI
llm = OpenAI()

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

system_template = "You are a {sports} historian."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)


chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt = chat_prompt.format_prompt(
    sports="F1", text="Who won the British Grand Prix 2015"
).to_messages()


from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI()
res = llm(chat_prompt)
print(res)
