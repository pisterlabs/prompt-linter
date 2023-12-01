from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from settings import config

OpenAI.api_key = config.OPENAI_API_KEY

llm = OpenAI(temperature=0.6, openai_api_key=OpenAI.api_key)

tool_names = ['wikipedia']
tools = load_tools(tool_names)
agent = initialize_agent(tools, llm, agent='zero-shot-react-description', verbose=True)


def wikipedia_question(question, self):
    print(f'Question: {question}')
    return agent.run(question)
