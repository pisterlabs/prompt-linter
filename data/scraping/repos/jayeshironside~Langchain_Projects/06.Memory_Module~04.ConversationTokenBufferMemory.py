#------------------------------
#ConversationTokenBufferMemory
#------------------------------
# ConversationTokenBufferMemory is a memory mechanism that stores recent interactions in a buffer within the system's memory. Unlike other methods that rely on the number of interactions, this memory system determines when to clear or flush interactions based on the length of tokens used. Tokens are units of text, like words or characters, and the buffer is cleared when the token count exceeds a certain threshold. By using token length as a criterion, the memory system ensures that the buffer remains manageable in terms of memory usage. This approach helps maintain efficient memory management and enables the system to handle conversations of varying lengths effectively.

# Pros of ConversationTokenBufferMemory
    # Efficient memory management: By using token length instead of the number of interactions, the memory system optimizes memory usage and prevents excessive memory consumption.
    # Flexible buffer size: The system adapts to conversations of varying lengths, ensuring that the buffer remains manageable and scalable.
    # Accurate threshold determination: Flushing interactions based on token count provides a more precise measure of memory usage, resulting in a better balance between memory efficiency and retaining relevant context.
    # Improved system performance: With efficient memory utilization, the overall performance of the system, including response times and processing speed, can be enhanced.
    
# Cons of ConversationTokenBufferMemory
    # Potential loss of context: Flushing interactions based on token length may result in the removal of earlier interactions that could contain important context or information, potentially affecting the accuracy of responses.
    # Complexity in threshold setting: Determining the appropriate token count threshold for flushing interactions may require careful consideration and experimentation to find the optimal balance between memory usage and context retention.
    # Difficulty in long-term context retention: Due to the dynamic nature of token-based flushing, retaining long-term context in the conversation may pose challenges as older interactions are more likely to be removed from the buffer.
    # Impact on response quality: In situations where high-context conversations are required, the token-based flushing approach may lead to a reduction in the depth of understanding and the quality of responses.

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
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationTokenBufferMemory(llm=llm, max_token_limit=30)) # Token limit is nothing but word limit

# print(conversation.prompt.template)
        # Current conversation:
        # {history}
        # Human: {input}
        # AI:

print(conversation.memory.prompt.template)
    # Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.

    # EXAMPLE
    # Current summary:
    # The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

    # New lines of conversation:
    # Human: Why do you think artificial intelligence is a force for good?
    # AI: Because artificial intelligence will help humans reach their full potential.

    # New summary:
    # The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
    # END OF EXAMPLE

    # Current summary:
    # {summary}

    # New lines of conversation:
    # {new_lines}

    # New summary:

print(conversation.predict(input="Good morning AI!"))
    # Current conversation:

    # Human: Good morning AI!
    # AI: Good morning! It's a beautiful day today, isn't it? How can I help you?

print(conversation.predict(input="My Name is sharath"))
    # Current conversation:
    # Human: Good morning AI!
    # AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
    # Human: My Name is sharath
    # AI: Nice to meet you, Sharath! Is there anything I can do for you today?

print(conversation.predict(input="I stay in hyderabad"))
    # Current conversation:
    # Human: My Name is sharath
    # AI:  Nice to meet you, Sharath! Is there anything I can do for you today?
    # Human: I stay in hyderabad
    # AI: Interesting! Hyderabad is a great city. What brings you to Hyderabad?

print(print(conversation.memory.buffer))
print(conversation.predict(input="What is my name?"))
# Output - I'm sorry, I don't know your name.