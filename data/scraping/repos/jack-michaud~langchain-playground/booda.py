from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.llms import OpenAI, OpenAIChat

from tools import add_memory, search_memory

load_dotenv()

# GPT-3.5
llm = OpenAI()

agent = initialize_agent(
    [add_memory, search_memory],
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    agent_kwargs={
        "prefix": "You are searching for underlying meaning in your memory. You have access to the following tools:",
    },
)

if __name__ == "__main__":
    while True:
        i = input("Begin rumination > ")
        print(agent(i))
