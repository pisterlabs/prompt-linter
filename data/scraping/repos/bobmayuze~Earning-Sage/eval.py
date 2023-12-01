from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI

from langsmith import Client
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.smith import RunEvalConfig, run_on_dataset

def construct_chain():
  # Replace with your own chain or agent
  llm = ChatOpenAI(temperature=0)
  return LLMChain.from_string(llm, "What's the answer to {query}")

client = Client()
eval_config = RunEvalConfig(
    evaluators=[
        RunEvalConfig.Criteria("relevance")
    ],
    eval_llm=ChatOpenAI(temperature=0),
)
chain_results = run_on_dataset(
    client,
    dataset_name="Apple_Earning_Call4",
    llm_or_chain_factory=construct_chain,
    evaluation=eval_config,
)
