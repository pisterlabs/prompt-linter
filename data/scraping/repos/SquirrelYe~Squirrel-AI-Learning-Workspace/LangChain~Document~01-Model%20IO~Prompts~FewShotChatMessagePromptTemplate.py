from langchain.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

examples = [
    {
        "input": "2+2",
        "output": "4"
    },
    {
        "input": "2+3",
        "output": "5"
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are wonderous wizard of math."),
    few_shot_prompt,
    ("human", "{input}"),
])

# 使用管道调用ChatOpenAI
chain = final_prompt | ChatOpenAI(temperature=0.0)
result1 = chain.invoke({"input": "100 + 200?"})

# 使用ChatOpenAI生成聊天回复
chatbot = ChatOpenAI(temperature=0.0)
result2 = chatbot.invoke(final_prompt.format(input="100 + 200?"))

print(final_prompt.format(input="100 + 200?"), '\n\n', result1, '\n\n', result2)