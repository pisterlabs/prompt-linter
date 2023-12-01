from langchain.llms import CTransformers
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent

# insert file path to your llm
llm_path = ""

llm = CTransformers(
    model = llm_path,
    model_type = "llama",
       max_new_tokens = 512,
    temperature = 0.7
)

agent = create_csv_agent(
    llm,
    "youtubers_df.csv",
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

agent.run("What can you tell me about this csv?")