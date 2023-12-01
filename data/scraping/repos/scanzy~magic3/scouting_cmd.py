"""Simple example of a chatbot (with memory) with command line interface."""

# please put your API key in os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# prompts setup
template = "You are a helpful assistant who generates suggestions for the user."

chat_prompt = ChatPromptTemplate(messages=[
    SystemMessagePromptTemplate.from_template(template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# note that return_messages=True is required for the memory to work
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# AI model setup
chain = ConversationChain(
    llm=ChatOpenAI(), prompt=chat_prompt, memory=memory)

# chat loop
while True:
    prompt = input("User: ")
    response = chain.run({"input": prompt})
    print("AI: ", response, "\n")