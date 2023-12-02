from pathlib import Path
import time
from typing import Optional, TypedDict

import easy_io

from llm_wrapper.utils import is_this_openai_model, is_this_model_for_chat
from openai_api_wrapper.cache_utils import read_cached_output, dump_output_to_cache


gpt_parameters: dict = {
    "model": "",
    "temperature": 0.,
    "max_tokens": 1024,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

palm_parameters: dict = {
    "model": '',
    "temperature": 0,
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "top_p": None,
    "top_k": None,
}

cohere_parameters: dict = {
    "model": '',
    "max_tokens": 512,
    "temperature": 0,
    "k": 0,
    "stop_sequences": [],
    "return_likelihoods": 'NONE'
}


class LlmApiOutput(TypedDict):
    prompt: str
    response: str


def llm_api(model_name: str, prompt: str, updated_parameters: dict={},
            overwrite_cache: bool=False, cache_dir: Path=Path("./llm_cache"),
            sleep_time: int=-1,
            add_output_string_for_non_chat_models: bool=True,
            openai_organization: Optional[str]=None) -> LlmApiOutput:
    """Call LLM APIs. You may set the parameters by using parameter_update. Output format is {"prompt": str, "response": str}. Cache will be stored in cache_dir.
    
    Args:
        sleep_time (int, optional): Sleep time in seconds. If the value is negative, the default value for each model is used. Defaults to -1."""
    
    # udpate prompts for non-chat llms
    if add_output_string_for_non_chat_models and not is_this_model_for_chat(model_name):
        prompt += "\n\nOutput:"
    
    parameters = {}
    if not is_this_openai_model(model_name):
        if model_name in ["claude-2"]:  # no parameters
            parameters = {"model_name": model_name}
        elif model_name == "text-bison-001":
            parameters = dict(palm_parameters, model=f"models/{model_name}")
        elif "command" in model_name:
            parameters = dict(cohere_parameters, model=model_name)
        else:
            raise ValueError(f"{model_name} is an invalid value for model_name argument.")
        
        parameters = dict(parameters, **updated_parameters)
    
    # read cache
    cached_output = {}
    if not overwrite_cache and not is_this_openai_model(model_name):
        # cache for openai models will be handled by openai_api_wrapper
        
        # for other models:
        cached_output = read_cached_output(parameters=parameters, prompt=prompt, cache_dir=cache_dir)
        
        # if cache is found, return the cache
        if len(cached_output) > 0:
            # Palm can output None. If the cache includes None, the cache will be ignored.
            if cached_output["response"] is None:
                cached_output = {}
            
            # if the above conditions are not applicable, the cache_output includes a propoer cache
            if len(cached_output) > 0:
                return cached_output
            # otherwise, the cache is ignored and the code will continue to call the api
    
    # llm api
    if is_this_openai_model(model_name):  # openai models
        from openai_api_wrapper.text_api import openai_text_api, get_chat_parameters
        
        # update parameters
        if "model" in updated_parameters:
            assert updated_parameters["model"] == model_name
        else:
            gpt_parameters["model"] = model_name
        updated_gpt_parameters = dict(gpt_parameters, **updated_parameters)
        
        from functools import partial
        openai_text_api_partially_filled = partial(openai_text_api,
                                                   cache_dir=cache_dir, overwrite_cache=overwrite_cache, organization=openai_organization,
                                                   sleep_time=1 if sleep_time < 0 else sleep_time)
        
        # call api
        if is_this_model_for_chat(model_name):  # chat models like gpt-4
            output = openai_text_api_partially_filled(mode="chat", parameters=get_chat_parameters(prompt=prompt, parameters=updated_gpt_parameters))
            response = output["response"]["choices"][0]["message"]["content"]
        else:  # completion models like text-davinci-003
            output = openai_text_api_partially_filled(mode="complete", parameters=dict(updated_gpt_parameters, prompt=prompt))
            response = output["response"]["choices"][0]["text"]
    else:
        # LLM APIs can return errors even when the input is valid (e.g. busy server).
        # To avoid the errors, the code will try to call the api multiple times.
        # If it hit the limit in loop_limit, the code will raise an error.
        if model_name == "text-bison-001":  # google palm
            import google.generativeai as palm
            
            # store your palm key in google_api_key.txt
            palm_key_path = Path("../google_api_key.txt")
            if not palm_key_path.exists():
                raise FileNotFoundError(f"google_api_key.txt is not found in {palm_key_path}. Please create the file and write your palm key in the file.")
            
            palm_key = easy_io.read_lines_from_txt_file(palm_key_path)[0]
            palm.configure(api_key=palm_key)
            
            loop_limit = 10
            for loop_count in range(loop_limit):
                try:
                    response = palm.generate_text(**dict(parameters, prompt=prompt)).result
                except Exception as e:
                    print(e)
                    if loop_count == loop_limit -1:
                        raise Exception(f"Received {loop_limit} Response Error for the same input from Palm. Please try again later.\nPrompt:\n{prompt}")
                    print("Wait for 10 seconds.")
                    time.sleep(10)
                    continue
                
                if response is None:
                    response = ""
                
                break
            
            # palm is limited to 30 requests per minute
            if sleep_time < 0:
                time.sleep(3)
            else:
                time.sleep(sleep_time)
        elif "command" in model_name:  # cohere models
            import cohere
            
            # store your cohere key in cohere_key.txt
            cohere_key_path = Path("../cohere_key.txt")
            if not cohere_key_path.exists():
                raise FileNotFoundError(f"cohere_key.txt is not found in {cohere_key_path}. Please create the file and write your cohere key in the file.")
            
            with open(cohere_key_path, "r") as f:
                cohere_key = f.read().strip()
            co = cohere.Client(cohere_key)
            
            loop_limit = 10
            for loop_count in range(loop_limit):
                try:
                    response = co.generate(**dict(parameters, prompt=prompt)).generations[0].text
                except Exception as e:
                    print(e)
                    if loop_count == loop_limit -1:
                        raise Exception(f"Received {loop_limit} Response Error for the same input from {model_name}. Please try again later.\nPrompt:\n{prompt}")
                    print("Wait for 60 seconds.")
                    time.sleep(60)
                    continue
                break

            # trial key is limited to 5 calls/min
            if sleep_time < 0:
                time.sleep(12)
            else:
                time.sleep(sleep_time)
        else:
            raise ValueError(f"model_name={model_name} is not implemented")

        # cache new output
        if len(cached_output) == 0:
            dump_output_to_cache(output_dict={"prompt": prompt, "response": response}, parameters=parameters, prompt=prompt)

    return {"prompt": prompt, "response": response}
