from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools
from langchain.llms import OpenAI
from tools import mermaid, search, recipe_search_engine
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True)
del load_dotenv

# for debugging
import langchain
langchain.debug = False


def main():
    llm = ChatOpenAI(temperature=0)
    tools = [
        Tool(
            name="mermaid_diagram_generator",
            func=mermaid,
            description="You can depict a situation using a Mermaid diagram, illustrating the scenario in visual form.",
        ),
        Tool(
            name="recipe_database_search_engine",
            func=recipe_search_engine,
            description="You can ask any cooking question and get the insight.",
        ),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # KR) 주어진 음식 또는 재료에 대한 taste vector 를 추론하시오. taste vector 는 대표적인 맛의 요소 5가지를 포함한다. 짠맛, 단맛, 쓴맛, 신맛, 고소함을 0 ~ 1 사이의 값으로 표현한다.
    # EN) Given a food or ingredient, infer the taste vector. The taste vector contains five representative taste elements. Salty, sweet, bitter, sour, and savory are expressed as values between 0 and 1.
    
    input = "chicken"
    taste_vector_template = f"\
        Given a food or ingredient, infer the taste vector. \
        The taste vector contains five representative taste elements. \
        Salty, sweet, bitter, sour, and savory are expressed as values between 0 and 1. \
        Please reference the recipe database search engine tool. \
        \n\n \
        - Input: {input} \n \
        - Output format: json \n \
        - Output example: {{'salty': 0.1, 'sweet': 0.2, 'bitter': 0.3, 'sour': 0.4, 'savory': 0.5}} \n \
        \n\n \
        Your final output must be a json format string only. "

    output = agent.run(taste_vector_template)
    print('\n>> Output')
    print(output)


if __name__ == "__main__":
    main()