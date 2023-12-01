import openai
from langchain import LangChain

# Set up OpenAI API credentials
openai.api_key = "YOUR_API_KEY"

# Create LangChain instance
chain = LangChain(prompt="Hello, I'm ChatGPT. How can I help you today?", max_turns=3)

# Define function to get OpenAI API response
def get_openai_response(prompt):
    response = openai.Completion.create(
        engine="davinci", prompt=prompt, max_tokens=1024, n=1, stop=None, temperature=0.7
    )
    return response.choices[0].text.strip()

# Chat loop
while True:
    user_input = input("You: ")
    response = chain(user_input, get_openai_response)
    print("ChatGPT:", response)
