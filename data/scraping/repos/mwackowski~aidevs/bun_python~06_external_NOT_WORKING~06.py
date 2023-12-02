from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from system_template import SYSTEM_TEMPLATE

human_template = "Q: {question}"

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        ("human", human_template),
    ]
)

formatted_chat_prompt = chat_prompt.format_messages(
    question="Today is October 13, 2023. What will the date after 193 days from now in the format MM/DD/YYYY?",
)

chat = ChatOpenAI(model="gpt-4")
content = chat.predict_messages(formatted_chat_prompt)

print(content)
cc = compile(content.content, "file", "eval")
print("Actual Date:", eval(cc))
