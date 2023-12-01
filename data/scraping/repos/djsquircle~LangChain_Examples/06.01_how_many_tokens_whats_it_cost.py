from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts import load_prompt
from langchain.callbacks import get_openai_callback

# Load each character's prompt
yasuke = load_prompt("./prompts/Yasuke.json")
lunavega = load_prompt("./prompts/LunaVega.json")
vitoprovolone = load_prompt("./prompts/VitoProvolone.json")

# Initialize the chat model
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=.95)

# Create a list of questions
questions = [
    "What's your opinion on the AI singularity?",
    "If you had to choose between wealth and power, which would you choose and why?",
    "If you were stranded on a deserted island, what would you do to survive?",
    "What are your thoughts on interstellar travel?",
    "If you could change one thing about the world, what would it be?",
    "What are your thoughts on the use of cryptocurrencies?",
    "How would you handle a situation where you've been betrayed by someone you trust?",
    "What's your most controversial opinion?",
    "How would you react if you found out you were being replaced by a newer, more advanced AI?",
    "What is your stance on the ethics of creating sentient AI?",
    "What would you do if you were put in charge of a country for a day?"
]

# Iterate over the questions
for question in questions:
    print(f"\nQuestion: {question}\n")

    # Create a chain for Yasuke and print the response
    with get_openai_callback() as cb:
        chain = LLMChain(llm=chat, prompt=yasuke)
        print(f"\nYasuke: {chain.run(question)}\n")
        print(cb)


    # Create a chain for Luna Vega and print the response
    with get_openai_callback() as cb:
        chain = LLMChain(llm=chat, prompt=lunavega)
        print(f"\nLuna Vega: {chain.run(question)}\n")
        print(cb)

    # Create a chain for Vito Provolone and print the response
    with get_openai_callback() as cb:
        chain = LLMChain(llm=chat, prompt=vitoprovolone)
        print(f"\nVito Provolone: {chain.run(question)}\n")
        print(cb)
