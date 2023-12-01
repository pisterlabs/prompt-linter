from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import dotenv

from dotenv import load_dotenv
load_dotenv()

llm = OpenAI( temperature = 0.9 )

tools = load_tools( ['google-search', 'Wikipedia', 'llm-math'],
                    llm = llm 
                )
agent = initialize_agent( tools, llm, 
                         agent = 'zero-shot-react-description'
                        )

agent.run ( 'Who was the lead actor in Interstellar? What is his current age raised to the power of 0.21? ')