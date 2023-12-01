from langchain.llms.google_palm import GooglePalm
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from dotenv import load_dotenv

load_dotenv()


def multiplier(a, b):
    return a * b


def parse_multiplier(s: str):
    """LLM calls this function with the input it decides.

    Args:
        s (str): LLM's input

    Returns:
        int: Muliplication of two numbers
    """
    a, b = s.split(",")
    return multiplier(int(a), int(b))


if __name__ == "__main__":
    llm = GooglePalm()  # type: ignore
    tools = [
        Tool(
            name="Multiplier",
            func=parse_multiplier,
            description="Useful for multiplying two numbers together. The input to this tool should be comma separated numbers you want multiply.",
        )
    ]

    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run("1233 times forty thousand sixty nine?")

    # the sky is the limit!
