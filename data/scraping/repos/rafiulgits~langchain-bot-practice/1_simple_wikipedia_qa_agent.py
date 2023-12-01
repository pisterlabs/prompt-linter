from langchain.llms import OpenAI
from langchain.agents import load_tools, AgentType, initialize_agent
from dotenv import load_dotenv

# loading environment variable, see .env file
load_dotenv(".env.local")

# temperature 0 for select the highest probability option
llm = OpenAI(temperature=0, verbose=False)
tools = load_tools(["wikipedia", "llm-math"],llm=llm)

# This agent uses the ReAct framework to determine which tool to use based solely on the tool's description
agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

question = ""
while True:
  question = input("[YOU]: ")
  if question.lower() == "quit":
    print("[BOT]: Good Bye")
    break
  try:
    answer = agent.run(question)
    print("[BOT]:", answer)
  except:
    print("[BOT]: Something went wrong")



# [YOU]: Where is Bangladesh?
# [BOT]: Bangladesh is located in South Asia, with a population of around 169 million people in an area of 148,460 square kilometres (57,320 sq mi). It shares land borders with India to the west, north, and east, and Myanmar to the southeast; to the south it has a coastline along the Bay of Bengal.
# [YOU]: What is the capital of that country?
# [BOT]: Something went wrong   

# second question has no context or simply agent has no previous reference in this codeblock