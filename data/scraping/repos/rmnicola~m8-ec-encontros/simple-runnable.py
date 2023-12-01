from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template(
"""
You are now my personal travel agent. Act as someone who has immense travel
experience and knows the best places in the world to do certain activities. I
want to know where I should go to {activity}. Give the answers as a list of
items, no bigger than 5 items. For each item, create a simple sentence
justifying this choice.
"""
)

chain = {"activity": RunnablePassthrough()} | prompt | model

for s in chain.stream("eat live fish"):
    print(s.content, end="", flush=True)
