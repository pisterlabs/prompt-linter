#-------------------------
# ConversationBufferMemory
#-------------------------
# Imagine you're having a conversation with someone, and you want to remember what you've discussed so far.  The ConversationBufferMemory does exactly that in a chatbot or similar system. It keeps a record, or "buffer," of the past parts of the conversation. 
    
# This buffer is an essential part of the context, which helps the chatbot generate better responses. The unique thing about this memory is that it stores the previous conversation exactly as they were, without any changes. It preserves the raw form of the conversation, allowing the chatbot to refer back to specific parts accurately. In summary, the ConversationBufferMemory helps the chatbot remember the conversation history, enhancing the overall conversational experience.

# Pros of ConversationBufferMemory
    # Complete conversation history: It retains the entire conversation history, ensuring comprehensive context for the chatbot.Accurate references: By storing conversation excerpts in their original form, it enables precise referencing to past interactions, enhancing accuracy.
    # Contextual understanding: The preserved raw form of the conversation helps the chatbot maintain a deep understanding of the ongoing dialogue.
    # Enhanced responses: With access to the complete conversation history, the chatbot can generate more relevant and coherent responses.
     
    
# Cons of ConversationBufferMemory:
    # Increased memory usage: Storing the entire conversation history consumes memory resources, potentially leading to memory constraints.
    # Potential performance impact: Large conversation buffers may slow down processing and response times, affecting the overall system performance.
    # Limited scalability: As the conversation grows, the memory requirements and processing load may become impractical for extremely long conversations.
    # Privacy concerns: Storing the entire conversation history raises privacy considerations, as sensitive or personal information may be retained in the buffer.

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
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

# print(conversation.prompt.template)
        # Current conversation:
        # {history}
        # Human: {input}
        # AI:

print(conversation.predict(input="Good Morning AI !"))
        # Current conversation:
        # Human: Good morning AI!
        # AI: Good morning! It's a beautiful day today, isn't it? How can I help you?

print(conversation.predict(input="My name is Jayesh"))
        # Current conversation:
        # Human: Good morning AI!
        # AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
        # Human: My name is Jayesh!
        # AI:

print(conversation.predict(input="I stay in Ahmedabad, India"))
        # Current conversation:
        # Human: Good morning AI!
        # AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
        # Human: My name is Jayesh!
        # AI:  Nice to meet you, Jayesh! Is there anything I can do for you today?
        # Human: I stay in Ahmedabad, India
        # AI: Ah, Ahmedabad! I've heard it's a great city. What do you like most about living there?

print(conversation.memory.buffer)
        # Human: Good morning AI!
        # AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
        # Human: My name is Jayesh!
        # AI:  Nice to meet you, Jayesh! Is there anything I can do for you today?
        # Human: I stay in Ahmedabad, India
        # AI:  Interesting! Ahmedabad is a beautiful city. What brings you to Ahmedabad?

print(conversation.predict(input="What is my name?"))
        # Output - Your name is jayesh