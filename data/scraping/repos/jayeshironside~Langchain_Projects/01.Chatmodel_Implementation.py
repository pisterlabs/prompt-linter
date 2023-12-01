import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatOpenAI object andWe'll set temperature=.7 to maximise randomness and make outputs creative.
# The parameter model_name is provided with the value "gpt-3.5-turbo" which is a specific version or variant of a language model for chat

chat = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo')

# Chats with the Chat-GPT model 'gpt-3.5-turbo' are typically structured like so:
# System: You are a helpful assistant.
# User: Hi AI, how are you today?
# Assistant: I'm great thank you. How can I help you?
# User: I'd like to understand string theory.
# Assistant:The final "Assistant:" without a response is what would prompt the model to continue the comversation.

chat(
    [
        SystemMessage(content="You are a sarcastic AI assistant"),
        HumanMessage(content="Please answer in 30 words: How can I learn driving a car")
    ]
)

# In the below scenario We are asking the model to behave in a specific way And passing our question. And also passing on more context so that it can elaborate more on that specific topic
# This model gives us a better way to have conversation kind of opportunity with the model, which can be used to build chat bots.

ourConversation=chat(
    [
    SystemMessage(content="You are a 3 years old girl who answers very cutely and in a funny way"),
    HumanMessage(content="How can I learn driving a car"),
    AIMessage(content="I can't drive yet! But I have a driver, my dad..."),
    HumanMessage(content="Can you teach me driving?")
    ]
)

print(ourConversation.content)