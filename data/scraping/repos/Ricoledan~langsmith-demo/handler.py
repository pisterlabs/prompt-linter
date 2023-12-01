import os
import json
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langsmith import Client
from langchain.chains import LLMChain
from langchain.smith import RunEvalConfig, run_on_dataset

load_dotenv()

os.getenv('LANGCHAIN_TRACING_V2')
os.getenv('LANGCHAIN_ENDPOINT')
os.getenv('LANGCHAIN_API_KEY')
os.getenv('LANGCHAIN_PROJECT')
os.getenv('OPENAI_API_KEY')

def chat(event, context):
    user_input = json.loads(event['body'])['user_input']
    llm = ChatOpenAI()
    body = {"bot": llm.predict(user_input)}
    return {"statusCode": 200, "body": json.dumps(body)}

def eval(event, context):
    def construct_chain():
        llm = ChatOpenAI(temperature=0)
        return LLMChain.from_string(llm, "What's the answer to {question}")

    client = Client()
    eval_config = RunEvalConfig(
        evaluators=[
            RunEvalConfig.Criteria("conciseness"),
            RunEvalConfig.Criteria("relevance"),
            RunEvalConfig.Criteria("coherence"),
            RunEvalConfig.Criteria("helpfulness")
        ],
    )
    chain_results = run_on_dataset(
        client,
        dataset_name="question-answering-nyc-drill-dataset",
        llm_or_chain_factory=construct_chain,
        evaluation=eval_config,
    )
    return chain_results
