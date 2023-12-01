from dotenv import load_dotenv
load_dotenv()

from langchain.utilities import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

db = SQLDatabase.from_uri("sqlite:///FlowerShop.db")
llm = OpenAI(temperature=0, verbose=True)

agent_executor = create_sql_agent(
  llm=llm,
  toolkit=SQLDatabaseToolkit(db=db, llm=llm),
  verbose=True,
  agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

questions = [ "哪种鲜花的存货数量最少？", "平均销售价格是多少？",]

for question in questions:
  response = agent_executor.run(question)
  print(response)