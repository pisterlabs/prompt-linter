from ctransformers import AutoModelForCausalLM
import openai

import config
import service_api_key
import util_time

local_llm = None
if config.is_local():
    gpu_message = f"Using {config.LOCAL_GPU_LAYERS} GPU layers" if config.IS_GPU_ENABLED else "NOT using GPU"
    print(f"LOCAL AI model: {config.LOCAL_MODEL_FILE_PATH} [{config.LOCAL_MODEL_TYPE}] [{gpu_message}]")
    local_llm = None
    if config.IS_GPU_ENABLED:
        local_llm = AutoModelForCausalLM.from_pretrained(config.LOCAL_MODEL_FILE_PATH, model_type=config.LOCAL_MODEL_TYPE, gpu_layers=config.LOCAL_GPU_LAYERS)
    else:
        local_llm = AutoModelForCausalLM.from_pretrained(config.LOCAL_MODEL_FILE_PATH, model_type=config.LOCAL_MODEL_TYPE)
else:
    print(f"Open AI model: {config.OPEN_AI_MODEL}]")
    openai.api_key = service_api_key.get_openai_key()

def get_completion_from_openai(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=config.OPEN_AI_MODEL,
        messages=messages,
        # Temperature is the degree of randomness of the model's output
        # 0 would be same each time. 0.7 or 1 would be difference each time, and less likely words can be used:
        temperature=config.TEMPERATURE,
    )
    return response.choices[0].message["content"]

def get_completion_from_local(prompt):
    return local_llm(prompt)

def get_completion(prompt):
    if config.is_local():
        return get_completion_from_local(prompt)
    else:
        return get_completion_from_openai(prompt)

def send_prompt(prompt, show_input = True, show_output = True):
    if show_input:
        print("=== INPUT ===")
        print(prompt)

    response = get_completion(prompt)

    if show_output:
        print("=== RESPONSE ===")
        print(response)

    return response

def next_prompt(prompt):
    start = util_time.start_timer()
    rsp = None
    if config.is_debug:
        rsp = send_prompt(prompt)
    else:
        rsp = send_prompt(prompt, False, False)
    elapsed_seconds = util_time.end_timer(start)
    return (rsp, elapsed_seconds)
