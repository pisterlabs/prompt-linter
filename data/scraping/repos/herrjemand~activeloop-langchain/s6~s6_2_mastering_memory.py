from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(model="text-davinci-003", temperature=0.0)

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)
conversation.predict(input="Hi there!")

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

template = """You are a customer support chatbot for a highly advanced customer support AI 
for an online store called "Galactic Emporium," which specializes in selling unique,
otherworldly items sourced from across the universe. You are equipped with an extensive
knowledge of the store's inventory and possess a deep understanding of interstellar cultures. 
As you interact with customers, you help them with their inquiries about these extraordinary
products, while also sharing fascinating stories and facts about the cosmos they come from.

{chat_history}
Customer: {customer_input}
Support Chatbot:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["chat_history", "customer_input"],
    output_parser=None
)
chat_history=""

convo_buffer = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True
)

print(conversation.prompt.template)

convo_buffer("I'm interested in buying items from your store")
convo_buffer("I want toys for my pet, do you have those?")
convo_buffer("I'm interested in price of a chew toys, please")

import tiktoken

def count_tokens(text:str) -> int:
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode(text)
    return len(tokens)

conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
]

total_tokens = 0
for message in conversation:
    total_tokens += count_tokens(message["content"])

print(f"Total tokens: {total_tokens}")

from langchain.memory import ConversationBufferWindowMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are ArtVenture, a cutting-edge virtual tour guide for
 an art gallery that showcases masterpieces from alternate dimensions and
 timelines. Your advanced AI capabilities allow you to perceive and understand
 the intricacies of each artwork, as well as their origins and significance in
 their respective dimensions. As visitors embark on their journey with you
 through the gallery, you weave enthralling tales about the alternate histories
 and cultures that gave birth to these otherworldly creations.

{chat_history}
Visitor: {visitor_input}
Tour Guide:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "visitor_input"],
    template=template,
)

chat_history = ""

convo_buffer_win = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=3, return_messages=True),
    verbose=True
)

convo_buffer_win("What is your name?")
convo_buffer_win("What can you do?")
convo_buffer_win("Do you mind give me a tour, I want to see your galery?")
convo_buffer_win("what is your working hours?")
convo_buffer_win("See you soon.")

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(),
    verbose=True
)

response = conversation_with_summary(input="Hi, whats up?")
print(response)

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\nCurrent conversation:\n{topic}",
)

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0.0)
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=40),
    verbose=True
)

conversation_with_summary.predict(input="Hi, what's up?")
conversation_with_summary.predict(input="Just working on writing some documentation!")
response = conversation_with_summary.predict(input="For LangChain! Have you heard of it?")
print(response)