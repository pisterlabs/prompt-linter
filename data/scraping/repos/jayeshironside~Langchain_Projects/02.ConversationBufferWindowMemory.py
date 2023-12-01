#------------------------------
#ConversationBufferWindowMemory
#------------------------------
# Imagine you have a limited space in your memory to remember recent conversations. The ConversationBufferWindowMemory is like having a short-term memory that only keeps track of the most recent interactions. It intentionally drops the oldest ones to make room for new ones. This helps manage the memory load and reduces the number of tokens used. The important thing is that it still keeps the latest parts of the conversation in their original form, without any modifications. So, it retains the most recent information for the chatbot to refer to, ensuring a more efficient and up-to-date conversation experience.

# Pros of ConversationBufferWindowMemory
    # Efficient memory utilization: It maintains a limited memory space by only retaining the most recent interactions, optimizing memory usage.
    # Reduced token count: Dropping the oldest interactions helps to keep the token count low, preventing potential token limitations. Unmodified context retention: The latest parts of the conversation are preserved in their original form, ensuring accurate references and contextual understanding.
    # Up-to-date conversations: By focusing on recent interactions, it allows the chatbot to stay current and provide more relevant responses.
    
# Cons of ConversationBufferWindowMemory:
    # Limited historical context: Since older interactions are intentionally dropped, the chatbot loses access to the complete conversation history, potentially impacting long-term context and accuracy.
    # Loss of older information: Valuable insights or details from earlier interactions are not retained, limiting the chatbot's ability to refer back to past conversations.
    # Reduced depth of understanding: Without the full conversation history, the chatbot may have a shallower understanding of the user's context and needs.
    # Potential loss of context relevance: Important information or context from older interactions may be disregarded, affecting the chatbot's ability to provide comprehensive responses in certain scenarios.

import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory
                                                  )
import tiktoken #Tiktoken, developed by OpenAI, is a tool used for text tokenization. Tokenization involves dividing a text into smaller units, such as letters or words. Tiktoken allows you to count tokens and estimate the cost of using the OpenAI API, which is based on token usage. It utilizes byte pair encoding (BPE), a compression algorithm that replaces frequently occurring pairs of bytes with a single byte.In summary, Tiktoken helps with efficient text processing, token counting, and cost estimation for using OpenAI's API.
from langchain.memory import ConversationTokenBufferMemory

llm = OpenAI(temperature=0, model_name='text-davinci-003')  #we can also use 'gpt-3.5-turbo'
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferWindowMemory(k=1)) # K will decide how much memory will the chat store.

# print(conversation.prompt.template)
        # Current conversation:
        # {history}
        # Human: {input}
        # AI:

print(conversation.predict(input="Good Morning AI !"))
        # Current conversation:
        # Human: Good morning AI!
        # AI: Good morning! It's a beautiful day today, isn't it? How can I help you?

print(conversation.predict(input="My Name is Jayesh"))
        # Current conversation:
        # Human: Good morning AI!
        # AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
        # Human: My Name is Jayesh
        # AI: Nice to meet you, Jayesh! Is there anything I can do for you today?

print(conversation.predict(input="I stay in Ahmedabad, India"))
        # Current conversation:
        # Human: My Name is sharath
        # AI:  Nice to meet you, Sharath! Is there anything I can do for you today?
        # Human: I stay in Ahmedabad, India
        # AI: Interesting! Ahmedabad is a beautiful city. What brings you to ahmedabad ?

print(conversation.memory.buffer) # BUffer will remember all the converasations

print(conversation.predict(input="What is my name?"))
# Output:- I'm sorry, I don't know your name. 
# This is because the K is set as 1 so the memory will remember only 1 past input rest of the data will be forgotten.