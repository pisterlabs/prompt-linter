from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_csv_agent, AgentType
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

def build_csv_agent(llm, file_path):
    assert isinstance(file_path, list)
    if len(file_path) == 1:
         file_path = file_path[0]
    
    csv_agent = create_csv_agent(
        llm,
        file_path,
        verbose=True,
        handle_parsing_errors=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    return csv_agent


def csv_as_tool(agent):
     return Tool.from_function(
                    name = "csv_retrieval_tool",
                    func= agent.run,
                    description= 'This tool useful for statistics, calculations, plotting and as well as data aggregation'
                )
"""while True:
    message = input('User:> ')
    try:
        response = 
    except ValueError as e:
        response = str(e)
        if not response.startswith("Could not parse tool input: "):
            raise e
        response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")

    print('Chatbot:> ', response)"""
