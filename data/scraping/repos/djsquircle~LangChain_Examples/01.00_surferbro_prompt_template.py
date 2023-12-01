# Import the necessary modules for handling environment variables and for working with LangChain.
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# First things first, let's load up the environment variables. 
# Why do we do this? Well, it's a best practice to keep sensitive information like API keys out of our code. 
# This way, we can share our code freely without worrying about exposing our secrets. 
# In our case, we'll be using an API key from OpenAI, which we'll load in the next steps.
load_dotenv()

# We're going to use the gpt-3.5-turbo model from OpenAI, and we're setting the temperature to 0. 
# Temperature affects the randomness of the AI's responses. A temperature of 0 makes the output completely deterministic, 
# providing the same output every time for a given input.
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Now we're going to create our chat templates. 
# These templates define the structure of the conversation, and we're going to use them to guide the AI's responses.

# The system message acts as the initial instruction for the AI. It sets the context for the conversation. 
# In our case, the AI is a helpful assistant that translates English to California surfer slang.
system_message_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant that translates English to California surfer slang.")

# We then add an example of a human message and an AI message. These are just examples to guide the AI in the conversation.
example_human = HumanMessagePromptTemplate.from_template("Hi")
example_ai = AIMessagePromptTemplate.from_template("What's up, dude?")

# We also specify a template for future human messages. In this case, it's just the text of the message.
human_message_prompt = HumanMessagePromptTemplate.from_template("{text}")

# We then create a chat prompt from all these templates. The chat prompt is what we will use to guide the AI in the conversation.
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, example_human, example_ai, human_message_prompt])

# We create a LangChain with our chat model and our chat prompt.
chain = LLMChain(llm=chat, prompt=chat_prompt)

# Finally, we run our chain with an example input, and print the result. In this case, the input is "I love programming."
# The AI will respond based on the templates we've given it and the input it receives.
print(chain.run("I love programming."))
