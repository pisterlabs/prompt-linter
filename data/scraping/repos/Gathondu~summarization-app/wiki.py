from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.utilities import WikipediaAPIWrapper

from utils import load_env


if __name__ == "__main__":
    load_env()

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    wikipedia = WikipediaAPIWrapper()
    tools = [
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Useful for when you need to get a summary from wikipedia about a topic",
        )
    ]
    agent_executor = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    print("Write Quit or Exit to exit.")
    i = 1
    while True:
        query = input(f"Question #{i}: ")
        if query.lower() in ["quit", "exit"]:
            import time

            print("Quitting.")
            time.sleep(2)
            break
        elif query == "":
            print("You must provide a question!")
            continue
        i += 1
        try:
            summary = agent_executor.run(query)
        except ValueError as e:
            summary = str(e)
            if not summary.startswith("Could not parse LLM output: `"):
                raise e
            summary = summary.removeprefix("Could not parse LLM output: `").removesuffix("`")
        print(f'SUMMARY \n {"-"*50} \n {summary} \n')
