from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI


def hunt_for_treasure(api_docs: str, st_callback: StreamlitCallbackHandler, model="gpt-4") -> str:
    llm = ChatOpenAI(model_name=model, temperature=0, streaming=True)
    tools = load_tools(["requests_all"])

    prefix = f"""
    You are on a 2D-grid and can see on the current and directly to adjacent cells. 
    Goal: You need to find and pick up the treasure. Specify only the command for a given situation.
    
    Call following API only using GET query-strings accordingly {api_docs}
    
    Rules of the game:
    - Cell descriptions: W=wall, G=ground, T=treausre
    - Available commands: UP DOWN LEFT RIGHT PICKUP_TREASURE
    - You can only pick up the treasure if the current cell has the treasure on it and is not a ground.
    - Do NOT attempt to pick up the treasure if it's in an adjacent cell. Wait until you have reached it.
    - Do NOT walk towards adjacent cells with walls. Wall-cells are marked by the letter 'W'.
    - Do NOT start the game again until you have found the treasure.
    
    Keep in mind:
    - Check your surroundings carefully to decide the next move.
    - Remember on what cells you have been and include it in your thoughts.
    - Solve the game step-by-step.
    - Don't repeat the same invalid moves.
    """
    agent = initialize_agent(tools,
                             llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             handle_parsing_errors=True,
                             verbose=True,
                             max_execution_time=90,
                             max_iterations=30)
    result = agent.run(prefix + "\nStart the game and find the treasure", callbacks=[st_callback])
    return result
