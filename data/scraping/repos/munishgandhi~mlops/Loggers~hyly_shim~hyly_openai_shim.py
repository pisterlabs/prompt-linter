import openai
from typing import Any, Dict, List, Optional
from datetime import datetime
import requests
import time
import os
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.organization = "org-kf0CNDw9Dvl65UAapwwoDpRF"
openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# Ashish - important!! Install mysql-connector-python instead of mysql-connector
import mysql.connector as SQLC


MODEL_COST_PER_1K_TOKENS = {
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-3.5-turbo": 0.002,
    "gpt-3.5-turbo-0301": 0.002,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-16k-0613": 0.003,
    "gpt-3.5-turbo-16k-0613-completion": 0.003,
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
}

def retry_on_error(max_retries=3, wait_time=2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                    print(f"Error occurred: {str(e)}")
                    retries += 1
                    print(f"Retrying... (Attempt {retries}/{max_retries})")
                    time.sleep(wait_time)
            raise Exception(f"Failed after {max_retries} retries.")
        return wrapper
    return decorator


def get_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    suffix = "-completion" if is_completion and (model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5-turbo-16k-0613")) else ""
    model = model_name.lower() + suffix
    if model not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model] * num_tokens / 1000

def record_into_mysql_org(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for):
    
    prompt_cost = get_openai_token_cost_for_model(model_name=model_name, num_tokens=prompt_tokens)
    completion_cost = get_openai_token_cost_for_model(model_name=model_name, 
                                                      num_tokens=completion_tokens,
                                                      is_completion=True)
    cost = prompt_cost+completion_cost
    cost = str(cost)
    db = SQLC.connect(
        host ="34.197.146.168",
        user ="mysqluser",
        password ="hyly",
        auth_plugin='mysql_native_password',
        database="hyly_openai"
        )
    print('connected')
    # Cursor to the database
    cursor = db.cursor()
    today = datetime.today().strftime('%Y-%m-%d')
    sql = "INSERT INTO openai_usage (date, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for, model_name, cost) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
    val = (today, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for, model_name, cost)
    cursor.execute(sql, val)
    db.commit()
    cursor.close()
    db.close()
    return prompt_cost, completion_cost, cost

def record_into_mysql(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for):
    prompt_cost = get_openai_token_cost_for_model(model_name=model_name, num_tokens=prompt_tokens)
    completion_cost = get_openai_token_cost_for_model(model_name=model_name, 
                                                      num_tokens=completion_tokens,
                                                      is_completion=True)
    cost = prompt_cost+completion_cost
    cost = str(cost)

    url = 'https://qa.hyly.us/graphql'
    headers = {'Authorization': 'userid___1709542253522009420'}
    body = f"""
       mutation {{
        createOpenaiHistory(
        input: {{
            gptModelName: "{model_name}",
            calledBy: "{who_is_using}",
            toolName: "{using_for}",
            totalPromptTokens: "{prompt_tokens}",
            totalCompletionTokens: "{completion_tokens}",
            totalTokens: "{total_tokens}",
            totalCost: "{cost}"
        }}
        ) {{
            openAiHistory {{
                gptModelName
                totalPromptTokens
                totalCompletionTokens
                totalTokens
                totalCost
                calledBy
                toolName
                promptTexts
                responseTexts
            }}
        }}
        }}

    """
    # print(body)
    response = requests.post(url=url, json={"query": body}, headers=headers)
    # print("response status code: ", response.status_code)
    # if response.status_code == 200:
        # print("response : ",response.content)
    return prompt_cost, completion_cost, cost

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def HylyOpenAICompletionFunctionOrg(message, functions, model_name, temperature,
                         who_is_using:str, app_category:str, app:str, app_field:str):
    """
    Usage:
    def chat(message):
        messages = []
        messages.append({"role": "user", "content": message})
        response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, total_tokens, total_cost, delay  = HylyOpenAICompletionFunction(
            message=messages,
            model_name = "gpt-3.5-turbo-0613",
            functions = functions,
            who_is_using = 'Rama',
            app_catetory = 'Langchain',
            app = 'Guest Card Tool',
            app_field = None
        )
        messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        
    """
    start = time.time()
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=message,
        temperature=temperature,
        functions = functions,
        function_call="auto"
    )
    end = time.time()
    prompt_tokens = response['usage'].prompt_tokens
    completion_tokens = response['usage'].completion_tokens
    total_tokens = response['usage'].total_tokens
    using_for = app
    delay = end - start
    
    pc, cc, tc = record_into_mysql(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for)
    print(f"Total Tokens: {total_tokens}, Total Cost: ${tc}")
    print(f"Prompt Tokens: {prompt_tokens}, Prompt Cost: ${pc}")
    print(f"Completion Tokens: {completion_tokens}, Completion Cost: ${cc}")
    print(f"API Time taken: {delay} seconds")
    return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay

@retry_on_error()
def HylyOpenAICompletionFunction(message, functions, model_name, temperature,
                         who_is_using:str, app_category:str, app:str, app_field:str):
    """
    Usage:
    def chat(message):
        messages = []
        messages.append({"role": "user", "content": message})
        response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, total_tokens, total_cost, delay  = HylyOpenAICompletionFunction(
            message=messages,
            model_name = "gpt-3.5-turbo-0613",
            functions = functions,
            who_is_using = 'Rama',
            app_catetory = 'Langchain',
            app = 'Guest Card Tool',
            app_field = None
        )
        messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        
    """
    start = time.time()
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"  # Replace with your actual API key
        }
        
        payload = {
            "model": model_name,
            "messages": message,
            "temperature": temperature,
            "functions": functions,
            "function_call": "auto"

        }
        # payload["functions"] = functions
        # payload["function_call"] = "auto"
        session = requests.Session()
        response = session.post(url, headers=headers, json=payload,timeout=30)

        if response.status_code == 200:
            response = response.json()
            # print(response)
            # usage = data["usage"]
            # response = data["choices"][0]["message"]["content"]
            # return response
            # return response, usage
        else:
            # Handle the API error here if needed
            response.raise_for_status()

        end = time.time()
        # print(response)
        prompt_tokens = response['usage']['prompt_tokens']
        completion_tokens = response['usage']['completion_tokens']
        total_tokens = response['usage']['total_tokens']
        using_for = app
        delay = end - start
        pc, cc, tc = record_into_mysql(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for)
        print(f"Total Tokens: {total_tokens}, Total Cost: ${tc}")
        print(f"Prompt Tokens: {prompt_tokens}, Prompt Cost: ${pc}")
        print(f"Completion Tokens: {completion_tokens}, Completion Cost: ${cc}")
        print(f"API Time taken: {delay} seconds")
        return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay
    except Exception as e:
        print(e)
        response = 'Sorry, could not get response'
        prompt_tokens =0 
        pc = 0
        completion_tokens =0
        cc = 0 
        total_tokens =0 
        tc = 0 
        delay = 0
        return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def HylyOpenAICompletionOrg(message, model_name, temperature, max_tokens,
                         who_is_using:str, app_category:str, app:str, app_field:str):
    """
    Usage:
    def chat(message):
        messages = []
        self.messages.append({"role": "user", "content": message})
        response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, total_tokens, total_cost, delay  = HylyOpenAICompletion(
            message=messages,
            model_name = "gpt-3.5-turbo-0613",
            temperature=0.0,
            max_tokens = 512,
            who_is_using = 'Rama',
            app_catetory = 'Langchain',
            app = 'Guest Card Tool',
            app_field = None
        )
        messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        
    """
    start = time.time()
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature
    )
    end = time.time()
    prompt_tokens = response['usage'].prompt_tokens
    completion_tokens = response['usage'].completion_tokens
    total_tokens = response['usage'].total_tokens
    using_for = app
    delay = end - start
    pc, cc, tc = record_into_mysql(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for)
    print(f"Total Tokens: {total_tokens}, Total Cost: ${tc}")
    print(f"Prompt Tokens: {prompt_tokens}, Prompt Cost: ${pc}")
    print(f"Completion Tokens: {completion_tokens}, Completion Cost: ${cc}")
    print(f"API Time taken: {delay} seconds")
    return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay

@retry_on_error()
async def HylyOpenAICompletion2(message, model_name, temperature, max_tokens,
                         who_is_using:str, app_category:str, app:str, app_field:str):
    """
    Usage:
    def chat(message):
        messages = []
        self.messages.append({"role": "user", "content": message})
        response, prompt_tokens, prompt_cost, completion_tokens, completion_cost, total_tokens, total_cost, delay  = HylyOpenAICompletion(
            message=messages,
            model_name = "gpt-3.5-turbo-0613",
            temperature=0.0,
            max_tokens = 512,
            who_is_using = 'Rama',
            app_catetory = 'Langchain',
            app = 'Guest Card Tool',
            app_field = None
        )
        messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        
    """
    start = time.time()

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"  # Replace with your actual API key
    }
    
    payload = {
        "model": model_name,
        "messages": message,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # if functions is not None and len(functions) > 0:
    #     payload["functions"] = functions
    #     payload["function_call"] = "auto"
    session = requests.Session()
    response = session.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        response = response.json()
        # print(response)
        # usage = data["usage"]
        # response = data["choices"][0]["message"]["content"]
        # return response
        # return response, usage
    else:
        # Handle the API error here if needed
        response.raise_for_status()

    end = time.time()
    # print(response)
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    total_tokens = response['usage']['total_tokens']
    using_for = app
    delay = end - start
    pc, cc, tc = record_into_mysql(model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using, using_for)
    print(f"Total Tokens: {total_tokens}, Total Cost: ${tc}")
    print(f"Prompt Tokens: {prompt_tokens}, Prompt Cost: ${pc}")
    print(f"Completion Tokens: {completion_tokens}, Completion Cost: ${cc}")
    print(f"API Time taken: {delay} seconds")
    return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay

@retry_on_error()
async def HylyOpenAICompletion(message, model_name, temperature, max_tokens,
                         who_is_using:str, app_category:str, app:str, app_field:str):
    
    task = asyncio.create_task(
        HylyOpenAICompletion2(message, model_name, temperature, max_tokens,
                         who_is_using, app_category, app, app_field)
    )
    try:
        async with asyncio.timeout(60):
            return await task
    except Exception as e:
        print(e)
        response = 'Sorry, could not get response'
        prompt_tokens =0 
        pc = 0
        completion_tokens =0
        cc = 0 
        total_tokens =0 
        tc = 0 
        delay = 0
        return response, prompt_tokens, pc, completion_tokens, cc, total_tokens, tc, delay
