from langchain.chat_models import ChatOpenAI

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

template = "You are a helpful assistant that translates {input_language} to {output_language}."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

msgs = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
print(msgs)

chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

output = chat_model.predict_messages(msgs)

print("output:", output)