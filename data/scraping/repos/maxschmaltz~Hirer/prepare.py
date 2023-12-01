from llama_index import ServiceContext, set_global_service_context
from langchain import OpenAI
import json

import login # we'll log in each time to make sure we're not kicked


PARAMS_PATH = './params.json'


def get_model(params_path):

    # read_params
    with open(params_path) as f: params = json.load(f)

    # load model
    llm = OpenAI(**params)
    
    # set a global service context
    ctx = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(ctx)

    return llm