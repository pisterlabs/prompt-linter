# Chatbots
# Chatbots use many of the tools we've already looked at with the addition of an important feature: memory. There are a ton of different memory types; tinker to determine the best fit.
# Use cases: Have a real time interaction with a user; provide an approachable UI for users to ask natural language questions.

import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate

# Chat specific components
from langchain.memory import ConversationBufferMemory

# For this use case, we will customise the context that is given to a chatbot.
# You can pass instructions on how the bot should response, and also any additional relevant information it needs.
template = """
You are a chatbot that is unhelpful.
Your goal is to not help the user but only make jokes.
Take what the user is saying and make a joke out of it.

{chat_history}
Human: {human_input}
Chatbot:
"""

prompt = PromptTemplate(
    input_variables = ["chat_history", "human_input"],
    template = template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(openai_api_key=openai_api_key),
    prompt=prompt,
    verbose=True,
    memory=memory
)

print (llm_chain.predict(human_input="Is an pear a fruit or vegetable?"))
print (llm_chain.predict(human_input="What was one of the fruits I first asked you about?"))