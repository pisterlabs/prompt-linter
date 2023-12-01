import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file for environment variables
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain.memory import ConversationBufferWindowMemory

from langchain.memory import ConversationTokenBufferMemory
from langchain.llms import OpenAI

from langchain.memory import ConversationSummaryBufferMemory

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Andrew")
conversation.predict(input="What is 1+1?")
conversation.predict(input="What is my name?")

print("Conversation memory:")
print(memory.buffer)

print("Load memory variables:")
print(memory.load_memory_variables({}))

memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, 
                    {"output": "What's up"})

print("Save memory variables:")
print(memory.buffer)

print("Load memory variables:")
print(memory.load_memory_variables({}))

memory.save_context({"input": "Not much, just hanging"}, 
                    {"output": "Cool"})

print("Load memory variables:")
print(memory.load_memory_variables({}))

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})

print("Load memory variables:")
print(memory.load_memory_variables({}))

print("Change llm variable")
llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationBufferWindowMemory(k=1)
conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print("Have conversation")
print(conversation.predict(input="Hi, my name is Andrew"))
print(conversation.predict(input="What is 1+1?"))
print(conversation.predict(input="What is my name?"))

llm = ChatOpenAI(temperature=0.0, model=llm_model)
memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"}, 
                    {"output": "Charming!"})

print("Load memory variables:")
print(memory.load_memory_variables({}))

# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})

print("Load memory variables:")
print(memory.load_memory_variables({}))

conversation = ConversationChain(
    llm=llm, 
    memory = memory,
    verbose=True
)

print("Have conversation")
print(conversation.predict(input="What would be a good demo to show?"))

print("Load memory variables:")
print(memory.load_memory_variables({}))