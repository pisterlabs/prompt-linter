import os
from operator import itemgetter
from dotenv import load_dotenv, find_dotenv
import openai

# Load environment variables
load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize LLM model
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

# Initialize memory
memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({"history": []})

# Define initial context
initial_context = "You are Rilla, a friendly chatbot."

# Define prompt template
prompt = ChatPromptTemplate.from_template(f"{initial_context}\n{{input}}")

# Initialize LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Print initial message
print("Rilla: Hello! I'm Rilla, your friendly chatbot.")

# Main conversation loop
while True:
    try:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Generate a response
        chatbot_response = chain.run(user_input)

        # Save the chat history
        memory.save_context({"input": user_input}, {"output": chatbot_response})

        # Output the chatbot's response
        print(f"Rilla: {chatbot_response}")

    except Exception as e:
        print(f"An error occurred: {e}")

# End the conversation
print("Rilla: Goodbye! Have a great day!")
