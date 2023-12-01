from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain import OpenAI, ConversationChain

llm = OpenAI(model="text-davinci-003", temperature=0.0)
conversation = ConversationChain(llm=llm)
output = conversation.predict(input="Hi there!")

print(output)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "Hi there!"},
    {"output": "Hi there! It's nice to meet you. How can I help you today?"},
)

print(memory.load_memory_variables({}))

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    memory = ConversationBufferMemory(),
)

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)

print(conversation.predict(input="Tell me a joke about elephants"))
print(conversation.predict(input="Who is the author of the Harry Potter series?"))
print(conversation.predict(input="What was the joke you told me earlier?"))

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

conversation = ConversationChain(
    llm = llm,
    verbose = True,
    memory = ConversationBufferMemory(),
)

user_message = "Tell me about the history of the Internet."
response = conversation(user_message)
print(response)

user_message = "Who are some important figures in its development?"
response = conversation(user_message)
print(response)  # Chatbot responds with names of important figures, recalling the previous topic

user_message = "What did Tim Berners-Lee contribute?"
response = conversation(user_message)
print(response)