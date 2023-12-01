#------------------------------
#ConversationBufferWindowMemory
#------------------------------
# With the ConversationBufferMemory, the length of the conversation keeps increasing, which can become a problem if it becomes too large for our LLM to handle. To overcome this, we introduce ConversationSummaryMemory. It keeps a summary of our past conversation snippets as our history. But how does it summarize? Here comes the LLM to the rescue! The LLM (Language Model) helps in condensing or summarizing the conversation, capturing the key information. So, instead of storing the entire conversation, we store a summarized version. This helps manage the token count and allows the LLM to process the conversation effectively. In summary, ConversationSummaryMemory keeps a condensed version of previous conversations using the power of LLM summarization.

# Pros of ConversationSummaryMemory
    # Efficient memory management- It keeps the conversation history in a summarized form, reducing the memory load.
    # Improved processing- By condensing the conversation snippets, it makes it easier for the language model to process and generate responses.
    # Avoiding maxing out limitations- It helps prevent exceeding the token count limit, ensuring the prompt remains within the processing capacity of the model.
    # Retains important information- The summary captures the essential aspects of previous interactions, allowing for relevant context to be maintained.

# Cons of ConversationSummaryMemory
    # Potential loss of detail: Since the conversation is summarized, some specific details or nuances from earlier interactions might be omitted.
    # Reliance on summarization quality: The accuracy and effectiveness of the summarization process depend on the language model's capability, which might introduce potential errors or misinterpretations.
    # Limited historical context: Due to summarization, the model's access to the complete conversation history may be limited, potentially impacting the depth of understanding.
    # Reduced granularity: The summarized form may lack the fine-grained information present in the original conversation, potentially affecting the accuracy of responses in certain scenarios.

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
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationSummaryMemory(llm=llm))

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

print(conversation.predict(input="My Name is Jayesh"))
    # Current conversation:

    # The human greeted the AI with a good morning, to which the AI responded with a good morning and asked how it could help. The human introduced themselves as Jayesh, to which the AI responded with a friendly greeting.
    # Human: My Name is Jayesh
    # AI: Hi Jayesh, it's nice to meet you. How can I help you today?

print(conversation.predict(input="I stay in Ahmedabad, India"))
    # Current conversation:

    # The human greeted the AI with a good morning, to which the AI responded with a good morning and asked how it could help. The human introduced themselves as Jayesh, to which the AI responded with a friendly greeting and asked how it could help.
    # Human: I stay in Ahmedabad, India
    # AI: Nice to meet you, Jayesh! It's great to hear that you live in Ahmedabad, India. Is there anything specific I can help you with today?

print(conversation.memory.buffer)
    # Output - The human greeted the AI with a good morning, to which the AI responded with a good morning and asked how it could help. The human introduced themselves as Jayesh and mentioned they live in Ahmedabad, India, to which the AI responded with a friendly greeting and asked how it could help.

print(conversation.predict(input="What is my name?"))
    # Current conversation:

    # The human greeted the AI with a good morning, to which the AI responded with a good morning and asked how it could help. The human introduced themselves as Jayesh and mentioned they live in Ahmedabad, India, to which the AI responded with a friendly greeting and asked how it could help. The AI then introduced itself as AI and asked what it could do for Jayesh.
    # Human: What is my name?
    # AI: Your name is Jayesh. Is there anything else I can help you with?