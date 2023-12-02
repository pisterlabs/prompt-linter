from llama_index import VectorStoreIndex
from llama_index.evaluation import QueryResponseEvaluator

import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index.evaluation import ResponseEvaluator
from llama_index import (
    LLMPredictor,
    ServiceContext,
)
from src.llama import chat
from pprint import pprint
from src.redis import rest



def prepare_evaluator(model_name="text-davinci-003"):
    # build service context
    apikey = os.environ["OPENAI_API_KEY"]
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name,streaming=True))
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    evaluator = QueryResponseEvaluator(service_context=service_context)
    return evaluator


def fetch_source_nodes(id):
    redis_res =  rest.SelectChat(id)
    source_nodes = [node_with_score.node.text for node_with_score in redis_res.source_nodes]
    return source_nodes

def query_evaluation(id,query):
    redis_res =  rest.SelectChat(id)
    evaluator = prepare_evaluator()
    eval_result = evaluator.evaluate_source_nodes(response=redis_res,query=query)
    return eval_result


if __name__ == "__main__":
    id = "8da4d957-d9e4-48f1-a2ff-ad717314f43a"
    query = "どのような人材を目指していますか 日本語で回答して"
    source_nodes = fetch_source_nodes(id)
    print(source_nodes)
    eval_result = query_evaluation(id,query)
    print(eval_result)