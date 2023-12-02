
from langchain.agents import load_tools, Tool 
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
import os 
# First load the language model we are going to use to control the agent.
llm = OpenAI(temperature=0, model_name="text-davinci-003")

search = SerpAPIWrapper
llm_math_chain = LLMMathChain(llm=llm)

tools = [
  Tool(
    name = "Search",
    func=search.run,
    description="useful for when you need to answer questions about current events."
  ),
  Tool(
    name="Music Search",
    func= lambda x: "'All I Want For Chistmas Is You' by Mariah Carey.",
    description="A Music search engine. Use this more often than the normal search if the question is about Music, like 'who is the singer of yesterday?' or 'what is the most popular song in 2023?'",
  ),
 Tool(
   name ="Calculator",
   func=llm_math_chain.run,
   description="useful for need to answer questions about math",
   return_direct=True
 ) 
]




#Next, let's load some tools to use. Note that the llm-math tool uses an LLM, so we need to pass that in.

tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
