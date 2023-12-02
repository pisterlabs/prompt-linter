from kani.engines.openai import OpenAIEngine
from kani.engines.huggingface.llama2 import LlamaEngine


# Fetching the proper kani engine for the specified model.
def generate_engine(**kwargs):
    assert kwargs['engine_name'] in ['openai', 'huggingface', 'llama', 'vicuna', 'ctransformers', 'llamactransformers'], "Specify a correct engine class name."
    if kwargs['engine_name'] == 'openai':
        api_key = input("Enter the API key for OpenAI API: ")
        engine = OpenAIEngine(api_key, model=kwargs['model_idx'])
    elif kwargs['engine_name'] == 'llama':
        engine = LlamaEngine(model_id=kwargs['model_idx'], use_auth_token=True)
        
    return engine
