import functools
import mlflow
import tempfile, json, os
from langchain.docstore.document import Document
import json, yaml
import os

langchain_patched = False

def serialize(obj):
    # Handle simple types
    if isinstance(obj, (int, float, str, bool)):
        return obj

    # Handle dicts
    elif isinstance(obj, dict):
        serialized_dict = {}
        for key, value in obj.items():
            if "key" in key.lower():
                serialized_dict[key] = "********"
            else:
                serialized_dict[key] = serialize(value)
        return serialized_dict

    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        serialized_list = []
        for item in obj:
            serialized_list.append(serialize(item))
        if isinstance(obj, tuple):
            return tuple(serialized_list)
        else:
            return serialized_list

    # Handle LangChain Document
    elif isinstance(obj, Document):
        serialized_doc = { 'page_content': obj.page_content, 'metadata': obj.metadata  } 
        return serialized_doc
    
    # handle pydantic models (openai uses pydantic)
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        return {str(obj.__class__): serialize(obj.dict())}
    
    # Handle other types
    else:
        return str(obj)

def log_json_artifact(json_data, artifact_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file in the temporary directory
        with open(os.path.join(temp_dir, artifact_name), "w") as f:
            json.dump(serialize(json_data), f)

        # Log the file as an artifact
        mlflow.log_artifact(os.path.join(temp_dir, artifact_name))

def log_yaml_artifact(yaml_data, artifact_name):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file in the temporary directory
        with open(os.path.join(temp_dir, artifact_name), "w") as f:
            yaml.dump(serialize(yaml_data), f)

        # Log the file as an artifact
        mlflow.log_artifact(os.path.join(temp_dir, artifact_name))

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global counter
        counter[-1] = counter[-1] + 1
        json_call = {}
        json_call['module'] = func.__module__
        json_call['name'] = func.__name__
        json_call['args'] = args
        json_call['kwargs'] = kwargs
        
        #print("json_call:", json_call)
        log_yaml_artifact(json_call, f"{'-'.join([str(i) for i in counter])}_{func.__module__}_{func.__name__}_call.yaml")
        counter.append(0)
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = e
            raise(e)
        finally:
            counter = counter[:-1]
            json_result = result
            #print("json_result:", json_result)
            log_yaml_artifact(json_result, f"{'-'.join([str(i) for i in counter])}-{func.__module__}_{func.__name__}_result.yaml")
        return result
    return wrapper

from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain import SerpAPIWrapper
from langchain.python import PythonREPL
from langchain import LLMMathChain


def patch_langchain():
    global counter
    counter = [0]
    global langchain_patched
    if not langchain_patched:
        langchain_patched = True
        AzureOpenAI._generate = log_function_call(AzureOpenAI._generate)
        AzureChatOpenAI._generate = log_function_call(AzureChatOpenAI._generate)
        SerpAPIWrapper.run = log_function_call(SerpAPIWrapper.run)
        PythonREPL.run = log_function_call(PythonREPL.run)
        LLMMathChain.run = log_function_call(LLMMathChain.run)

    