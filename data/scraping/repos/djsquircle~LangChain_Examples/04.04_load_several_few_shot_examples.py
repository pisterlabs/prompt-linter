from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import load_prompt

# Load each character's prompt
yasuke = load_prompt("./prompts/Yasuke.json")
lunavega = load_prompt("./prompts/LunaVega.json")
vitoprovolone = load_prompt("./prompts/VitoProvolone.json")

# Initialize the chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a list of questions
questions = ["What's the secret to happiness?", "What does honor mean to you?", "How do you approach conflicts?", "What should I do if my passport expires in Costa Rica and I can't get on the plane home?"]

# Iterate over the questions
for question in questions:
    print(f"\nQuestion: {question}")

    # Create a chain for Yasuke and print the response
    chain = LLMChain(llm=chat, prompt=yasuke)
    print(f"Yasuke: {chain.run(question)}")

    # Create a chain for Luna Vega and print the response
    chain = LLMChain(llm=chat, prompt=lunavega)
    print(f"Luna Vega: {chain.run(question)}")

    # Create a chain for Vito Provolone and print the response
    chain = LLMChain(llm=chat, prompt=vitoprovolone)
    print(f"Vito Provolone: {chain.run(question)}")
