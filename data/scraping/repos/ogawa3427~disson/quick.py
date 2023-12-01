from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memories import LocalMemory  # Correct import for LocalMemory
import langchain
import os

# Assuming 'OPENAI_API_KEY' is set in your environment variables, this will work.
# Otherwise, you need to set it manually in your script, but be careful with security.
openai.api_key = os.getenv('OPENAI_API_KEY')

# Create a LocalMemory object to store the chat history.
memory = LocalMemory("chat_history.json")

# Initialize LLMChain. You can specify your GPT model here.
# Note: The model_name should match one of the available models.
# "gpt-4-1106-preview" might not be a valid model name.
# You should replace it with a valid model name such as "gpt-4".
chain = LLMChain(model_name="gpt-4")

# Define the chat function.
def chat():
    while True:
        # Get user input.
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Retrieve the past conversation history.
        context = memory.get_context()

        # Ask the model a question and generate a response.
        # Note: run() should be replaced with the appropriate method to get a response.
        # This might be something like chain.generate() or chain.run_chain().
        response = chain.generate(prompt=user_input, context=context)  # Adjusted method name

        # Display the response.
        print("Bot:", response)

        # Update the conversation history.
        memory.update(user_input, response)

if __name__ == '__main__':
    chat()
